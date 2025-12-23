import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from pathlib import Path
from functools import partial
import copy
from torch.optim import Adam
import torchvision.utils as utils
import numpy as np
import wandb
import os
from datasets import load_dataset

# Import from the existing package
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    GaussianDiffusion, 
    Trainer, 
    Dataset, 
    Dataset_Aug1, 
    extract, 
    cosine_beta_schedule, 
    default, 
    cycle, 
    loss_backwards
)

class HFDataset(data.Dataset):
    def __init__(self, dataset_name, image_size, split='train'):
        super().__init__()
        try:
            self.dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            # Try loading with streaming if it's huge, or just fail
            print(f"Error loading dataset {dataset_name}: {e}")
            raise e
            
        self.image_size = image_size
        
        # Determine image column name (simple heuristic)
        self.image_key = 'image'
        if 'image' not in self.dataset.features:
             # Fallback or search for Image feature
             for key, feature in self.dataset.features.items():
                 # Check if the feature is an Image type (needs datasets.Image) or just guess by name
                 if key in ['img', 'image', 'picture', 'file']:
                     self.image_key = key
                     break
        
        print(f"Using column '{self.image_key}' as image source for {dataset_name}")

        self.transform = utils.transforms.Compose([
            utils.transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            utils.transforms.RandomCrop(image_size),
            utils.transforms.RandomHorizontalFlip(),
            utils.transforms.ToTensor(),
            utils.transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img = item[self.image_key]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

class DriftDiffusion(GaussianDiffusion):
    def __init__(
        self, 
        denoise_fn, 
        *, 
        image_size, 
        channels=3, 
        timesteps=1000, 
        loss_type='l1', 
        train_routine='Final', 
        sampling_routine='default', 
        beta_schedule1='cosine', 
        beta_schedule2='linear'
    ):
        super().__init__(
            denoise_fn, 
            image_size=image_size, 
            channels=channels, 
            timesteps=timesteps, 
            loss_type=loss_type, 
            train_routine=train_routine, 
            sampling_routine=sampling_routine
        )
        
        # Define Schedule 1
        if beta_schedule1 == 'cosine':
            betas1 = cosine_beta_schedule(timesteps)
        elif beta_schedule1 == 'linear':
            betas1 = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule1}")
            
        alphas1 = 1. - betas1
        alphas_cumprod1 = torch.cumprod(alphas1, axis=0)
        
        self.register_buffer('alphas_cumprod1', alphas_cumprod1)
        self.register_buffer('sqrt_alphas_cumprod1', torch.sqrt(alphas_cumprod1))
        self.register_buffer('sqrt_one_minus_alphas_cumprod1', torch.sqrt(1. - alphas_cumprod1))

        # Define Schedule 2
        if beta_schedule2 == 'cosine':
            betas2 = cosine_beta_schedule(timesteps)
        elif beta_schedule2 == 'linear':
            betas2 = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule2}")
            
        alphas2 = 1. - betas2
        alphas_cumprod2 = torch.cumprod(alphas2, axis=0)
        
        self.register_buffer('alphas_cumprod2', alphas_cumprod2)
        self.register_buffer('sqrt_alphas_cumprod2', torch.sqrt(alphas_cumprod2))
        self.register_buffer('sqrt_one_minus_alphas_cumprod2', torch.sqrt(1. - alphas_cumprod2))
        
    def q_sample_mixed(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        b = x_start.shape[0]
        half = b // 2
        
        # Ensure batch size is even and matches
        assert b % 2 == 0, "Batch size must be even for DriftDiffusion"
        
        # Split inputs
        x1 = x_start[:half]
        x2 = x_start[half:]
        t1 = t[:half]
        t2 = t[half:]
        n1 = noise[:half]
        n2 = noise[half:]
        
        # Process first half with schedule 1
        out1 = (
            extract(self.sqrt_alphas_cumprod1, t1, x1.shape) * x1 +
            extract(self.sqrt_one_minus_alphas_cumprod1, t1, x1.shape) * n1
        )
        
        # Process second half with schedule 2
        out2 = (
            extract(self.sqrt_alphas_cumprod2, t2, x2.shape) * x2 +
            extract(self.sqrt_one_minus_alphas_cumprod2, t2, x2.shape) * n2
        )
        
        return torch.cat([out1, out2], dim=0)

    def p_losses(self, x_start, x_end, t):
        # x_end is passed as noise from the trainer
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_mix = self.q_sample_mixed(x_start=x_start, t=t, noise=x_end)
            x_recon = self.denoise_fn(x_mix, t)
            
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()
                
            return loss, x_mix, x_recon
        
        return 0, None, None

    def forward(self, x1, x2, *args, **kwargs):
        # x1 is image batch, x2 is noise batch
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x1, x2, t, *args, **kwargs)

class DriftTrainer(Trainer):
    def __init__(
        self,
        diffusion_model,
        folder1,
        folder2,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        load_path = None,
        shuffle=True
    ):
        # Call super init mostly to set attributes, but we will overwrite datasets/dataloaders
        super().__init__(
            diffusion_model,
            folder1, # placeholder
            ema_decay=ema_decay,
            image_size=image_size,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            train_num_steps=train_num_steps,
            gradient_accumulate_every=gradient_accumulate_every,
            fp16=fp16,
            step_start_ema=step_start_ema,
            update_ema_every=update_ema_every,
            save_and_sample_every=save_and_sample_every,
            results_folder=results_folder,
            load_path=load_path,
            dataset='train', # Force use of Dataset_Aug1 logic if we used super(), but we will override
            shuffle=shuffle
        )
        
        # Initialize WandB
        wandb.init(project="drift_diffusion", config={
            "model": "DriftDiffusion",
            "image_size": image_size,
            "train_batch_size": train_batch_size,
            "train_num_steps": train_num_steps,
            "folder1": folder1,
            "folder2": folder2
        })

        # Setup Datasets
        # We use Dataset_Aug1 for training as in the reference script
        self.ds1 = Dataset_Aug1(folder1, image_size)
        self.ds2 = Dataset_Aug1(folder2, image_size)
        
        assert train_batch_size % 2 == 0, "Batch size must be even"
        half_batch = train_batch_size // 2
        
        self.dl1 = cycle(data.DataLoader(self.ds1, batch_size = half_batch, shuffle=shuffle, pin_memory=True, num_workers=4, drop_last=True))
        self.dl2 = cycle(data.DataLoader(self.ds2, batch_size = half_batch, shuffle=shuffle, pin_memory=True, num_workers=4, drop_last=True))
        
    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                # Load half batches
                data_1_half = next(self.dl1)
                data_2_half = next(self.dl2)
                
                # Concatenate to form full batch [Path1_Images, Path2_Images]
                data_1 = torch.cat([data_1_half, data_2_half], dim=0)
                
                # Generate noise
                data_2 = torch.randn_like(data_1)

                data_1, data_2 = data_1.cuda(), data_2.cuda()
                
                # Forward pass returns loss, and for visualization we might want more info, 
                # but p_losses returns (loss, x_mix, x_recon)
                loss, x_mix, x_recon = self.model(data_1, data_2)
                
                # Check if we need to log images (on the first accumulation step of the save interval)
                if self.step % self.save_and_sample_every == 0 and i == 0:
                    self.log_images(data_1, x_mix, x_recon)

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
                acc_loss = acc_loss/(self.save_and_sample_every+1)
                print(f'Mean of last {self.step}: {acc_loss}')
                acc_loss=0

                self.save()
                
            self.step += 1

        print('training completed')
        wandb.finish()

    def log_images(self, original, noised, reconstructed):
        # Unnormalize
        original = (original + 1) * 0.5
        noised = (noised + 1) * 0.5
        reconstructed = (reconstructed + 1) * 0.5
        
        # Take a few samples (e.g., 4 from first half, 4 from second half)
        b = original.shape[0]
        half = b // 2
        indices = list(range(0, min(4, half))) + list(range(half, half + min(4, half)))
        
        images_to_log = []
        for idx in indices:
            caption = "Path 1" if idx < half else "Path 2"
            
            # Combine side by side
            combined = torch.cat([original[idx], noised[idx], reconstructed[idx]], dim=2) # Concat along width
            
            # Convert to PIL or numpy for wandb
            # combined is (C, H, W*3)
            img_np = combined.cpu().detach().permute(1, 2, 0).numpy()
            images_to_log.append(wandb.Image(img_np, caption=f"{caption} (Orig | Noisy | Recon)"))
            
        wandb.log({"examples": images_to_log}, step=self.step)
