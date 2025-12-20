import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import numpy as np
import copy
from functools import partial
import wandb
import os
from PIL import Image
from inspect import isfunction

try:
    from apex import amp
except:
    pass

# Helper functions and classes duplicated from demixing_diffusion_pytorch.py to avoid circular imports

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower().lstrip('.') in exts]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

# Main classes

class SICDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        k_steps = 5, # Total number of sources (K)
        loss_type = 'l1'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.k_steps = k_steps
        self.loss_type = loss_type

    def q_sample(self, x_seq, alphas, t):
        """
        Forward process: Superimposing images.
        
        x_seq: (B, K, C, H, W) - Sequence of images sorted by strength. x_seq[:, 0] is the target.
        alphas: (B, K) - Weights for each image.
        t: (B,) - Current time step (0 to K-1). Indicates how many interference layers to add.
        
        Returns y^{(t)} = x^{(1)} + sum_{i=2}^{t+1} alpha_i x^{(i)}
        """
        B, K, C, H, W = x_seq.shape
        device = x_seq.device
        
        # y^{(0)} = x^{(1)}
        y = x_seq[:, 0].clone()
        
        # We need to add interference terms.
        # x^{(i)} corresponds to x_seq[:, i-1] in 0-indexing.
        # summation is for i from 2 to t+1.
        # in 0-indexing, indices from 1 to t.
        
        # Create mask for indices 1 to K-1
        j_indices = torch.arange(1, K, device=device) # [1, 2, ..., K-1]
        
        # Mask: we include index j if j <= t
        # shape (1, K-1) broadcast to (B, K-1)
        mask = j_indices[None, :] <= t[:, None]
        
        x_interferences = x_seq[:, 1:] # (B, K-1, C, H, W)
        alpha_interferences = alphas[:, 1:] # (B, K-1)
        
        weighted_interferences = x_interferences * alpha_interferences[:, :, None, None, None]
        
        # Zero out terms that shouldn't be included
        masked_interferences = weighted_interferences * mask[:, :, None, None, None].float()
        
        # Sum over the K dimension
        total_interference = masked_interferences.sum(dim=1)
        
        return y + total_interference

    def p_losses(self, x_seq, alphas, t):
        """
        Compute loss for predicting the top interference layer.
        
        t ranges from 1 to K-1.
        We predict x^{(t+1)} given y^{(t)}.
        """
        # Target is x^{(t+1)}, which is x_seq[:, t]
        target = x_seq[torch.arange(x_seq.shape[0]), t] # (B, C, H, W)
        
        # Input is y^{(t)}
        y_t = self.q_sample(x_seq, alphas, t)
        
        # Conditioning: alpha_{t+1} corresponds to alphas[:, t]
        alpha_cond = alphas[torch.arange(alphas.shape[0]), t]
        
        # Prepare input for denoise_fn
        # We append alpha as an extra channel
        b, c, h, w = y_t.shape
        alpha_map = alpha_cond[:, None, None, None].expand(b, 1, h, w)
        model_input = torch.cat((y_t, alpha_map), dim=1)
        
        # Predict
        prediction = self.denoise_fn(model_input, t)
        
        if self.loss_type == 'l1':
            loss = (target - prediction).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, prediction)
        else:
            raise NotImplementedError()
            
        return loss

    def forward(self, x_seq, alphas):
        """
        x_seq: (B, K, C, H, W)
        alphas: (B, K)
        """
        b = x_seq.shape[0]
        device = x_seq.device
        
        # Sample t from 1 to K-1
        # t=0 is not trained on because we don't predict anything from y^{(0)} in the reverse loop 
        # (reverse loop stops at getting y^{(0)})
        
        if self.k_steps <= 1:
            return torch.tensor(0., device=device, requires_grad=True)
            
        t = torch.randint(1, self.k_steps, (b,), device=device).long()
        
        return self.p_losses(x_seq, alphas, t)

    @torch.no_grad()
    def sample(self, y_final, alphas):
        """
        Perform SIC to recover the clean signal.
        
        y_final: (B, C, H, W) - The observation y^{(K-1)}
        alphas: (B, K) - Known weights
        """
        y = y_final
        b, c, h, w = y.shape
        device = y.device
        
        # Iteratively remove interference
        # From t = K-1 down to 1
        for t_val in reversed(range(1, self.k_steps)):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            
            # alpha_{t+1} is alphas[:, t_val]
            alpha_cond = alphas[:, t_val]
            
            # Input
            alpha_map = alpha_cond[:, None, None, None].expand(b, 1, h, w)
            model_input = torch.cat((y, alpha_map), dim=1)
            
            # Predict interference x^{(t+1)}
            x_pred = self.denoise_fn(model_input, t)
            
            # Remove interference
            # y^{(t-1)} = y^{(t)} - alpha_{t+1} * x^{(t+1)}
            y = y - alpha_cond[:, None, None, None] * x_pred
            
        return y

class SICTrainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        load_path = None,
        shuffle = True,
        image_size = 128,
        wandb_project = 'SIC_Diffusion',
        wandb_run_name = None
    ):
        super().__init__()
        self.model = diffusion_model
        self.k_steps = diffusion_model.k_steps
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        
        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        self.ds = Dataset(folder, image_size)
        # We need to fetch enough images to form batches of size (B, K)
        # The dataloader will return flat batches, we will reshape them in training loop
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size * self.k_steps, shuffle = shuffle, pin_memory = True, num_workers = 4, drop_last = True))
        
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)
        self.step = 0
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        
        self.fp16 = fp16
        
        self.reset_parameters()
        
        if load_path != None:
            self.load(load_path)
            
        if wandb_run_name:
            wandb.init(project=wandb_project, name=wandb_run_name)
        else:
            wandb.init(project=wandb_project)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def prepare_batch(self, batch):
        # batch: (B*K, C, H, W)
        device = batch.device
        B = self.batch_size
        K = self.k_steps
        C, H, W = batch.shape[1:]
        
        # Reshape to (B, K, C, H, W)
        x_seq_raw = batch.view(B, K, C, H, W)
        
        # Designate index 0 as target x^{(1)}
        targets = x_seq_raw[:, 0] # (B, C, H, W)
        interferences = x_seq_raw[:, 1:] # (B, K-1, C, H, W)
        
        # Generate random weights for interferences
        # alpha_i in (0, 1]
        alpha_interferences = torch.rand((B, K-1), device=device) * 0.9 + 0.1 # Avoid too small weights? Or just rand
        
        # Calculate weighted strength
        weighted_interferences = interferences * alpha_interferences[:, :, None, None, None]
        norms = weighted_interferences.flatten(2).norm(dim=2) # (B, K-1)
        
        # Sort interferences by strength descending
        sorted_indices = torch.argsort(norms, dim=1, descending=True) # (B, K-1)
        
        # Gather sorted alphas
        sorted_alpha_interferences = torch.gather(alpha_interferences, 1, sorted_indices)
        
        # Gather sorted interferences
        # Need to expand indices for gather
        # sorted_indices shape (B, K-1) -> (B, K-1, C, H, W)
        sorted_indices_expanded = sorted_indices[:, :, None, None, None].expand(-1, -1, C, H, W)
        sorted_interferences = torch.gather(interferences, 1, sorted_indices_expanded)
        
        # Construct final x_seq and alphas
        # x_seq: concat target and sorted interferences
        x_seq = torch.cat((targets.unsqueeze(1), sorted_interferences), dim=1)
        
        # alphas: concat 1.0 and sorted alpha_interferences
        ones = torch.ones((B, 1), device=device)
        alphas = torch.cat((ones, sorted_alpha_interferences), dim=1)
        
        return x_seq, alphas

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                x_seq, alphas = self.prepare_batch(data)
                
                loss = self.model(x_seq, alphas)
                
                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
                    wandb.log({"loss": loss.item()}, step=self.step)
                    
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                # Validation / Sampling
                self.ema_model.eval()
                with torch.no_grad():
                    # Get a fresh batch
                    val_data = next(self.dl).cuda()
                    x_seq_val, alphas_val = self.prepare_batch(val_data)
                    
                    # Construct full superposition y^{(K-1)}
                    # t = K-1
                    t_full = torch.full((self.batch_size,), self.k_steps - 1, device=x_seq_val.device, dtype=torch.long)
                    y_full = self.ema_model.module.q_sample(x_seq_val, alphas_val, t_full) # Use module for DataParallel
                    
                    # Reconstruct
                    reconstructed = self.ema_model.module.sample(y_full, alphas_val)
                    
                    target = x_seq_val[:, 0]
                    
                    # Prepare images for wandb
                    # Unnormalize from [-1, 1] to [0, 1]
                    target_vis = (target + 1) * 0.5
                    y_full_vis = (y_full + 1) * 0.5
                    reconstructed_vis = (reconstructed + 1) * 0.5
                    
                    # Grid
                    grid_target = utils.make_grid(target_vis, nrow=4)
                    grid_input = utils.make_grid(y_full_vis, nrow=4)
                    grid_recon = utils.make_grid(reconstructed_vis, nrow=4)
                    
                    wandb.log({
                        "Target (Clean)": wandb.Image(grid_target),
                        "Input (Superimposed)": wandb.Image(grid_input),
                        "Reconstructed": wandb.Image(grid_recon)
                    }, step=self.step)
                
                self.save()
                self.ema_model.train()

            self.step += 1

        print('training completed')
        wandb.finish()
