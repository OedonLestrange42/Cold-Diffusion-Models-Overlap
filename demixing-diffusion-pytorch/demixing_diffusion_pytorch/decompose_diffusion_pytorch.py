import math
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
from functools import partial
from inspect import isfunction
from einops import rearrange
from tqdm import tqdm
from Fid.fid_score import calculate_fid_given_samples
import errno
import os
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

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

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

class DecomposeDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        num_sources=4
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.num_sources = int(num_sources)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def k_of_t(self, t):
        k = 1 + torch.floor((self.num_sources - 1) * (t.float() + 1) / self.num_timesteps).long()
        return torch.clamp(k, min=1, max=self.num_sources)

    def normalize_sum(self, x_sum, k):
        k_clamped = torch.clamp(k, min=1)
        k_view = k_clamped.view(-1, 1, 1, 1).type_as(x_sum)
        out = x_sum / k_view
        return torch.clamp(out, -1.0, 1.0)

    def q_sample_sum(self, xs, t):
        b = xs[0].shape[0]
        x_sum = torch.zeros_like(xs[0])
        for i in range(len(xs)):
            x_sum = x_sum + xs[i]
        k = torch.full((b,), len(xs), dtype=torch.long, device=xs[0].device)
        xt = self.normalize_sum(x_sum, k)
        return xt

    def p_losses(self, xs, t):
        x_start = xs[0]
        x_mix = self.q_sample_sum(xs=xs, t=t)
        x_recon = self.denoise_fn(x_mix, t)
        if self.loss_type == 'l1':
            loss = (x_start - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, x_recon)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, xs, *args, **kwargs):
        b, c, h, w, device, img_size = *xs[0].shape, xs[0].device, self.image_size
        assert h == img_size and w == img_size
        t = kwargs.get('t', None)
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(xs, t, *args, **kwargs)

    @torch.no_grad()
    def all_sample(self, batch_size=16, xs=None, times=None):
        self.denoise_fn.eval()
        t = self.num_timesteps if times is None else times
        X1_0s, X_ts = [], []
        for step_val in range(t, 0, -1):
            step = torch.full((batch_size,), step_val - 1, dtype=torch.long, device=xs[0].device)
            x_mix = self.q_sample_sum(xs=xs, t=step)
            x1_bar = self.denoise_fn(x_mix, step)
            X1_0s.append(x1_bar.detach().cpu())
            X_ts.append(x_mix.detach().cpu())
        return X1_0s, X_ts

class Dataset_Aug(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        allowed_exts = set(['.' + e.lower() for e in exts])
        self.paths = [p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in allowed_exts]
        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
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

class Dataset_Center(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        allowed_exts = set(['.' + e.lower() for e in exts])
        self.paths = [p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in allowed_exts]
        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
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

class DecomposeTrainer(object):
    def __init__(
        self,
        diffusion_model,
        folders,
        *,
        ema_decay=0.995,
        image_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results_decompose',
        dataset='train',
        shuffle=True,
        use_wandb=False,
        wandb_project='DecomposeDiffusion',
        wandb_run_name=None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.fp16 = fp16
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_run = None
        if dataset == 'train':
            ds_cls = Dataset_Aug
        else:
            ds_cls = Dataset_Center
        self.datasets = [ds_cls(folder, image_size) for folder in folders]
        if len(self.datasets) != 2:
            raise ValueError('Require exactly two data_paths: target first, interference second.')
        for ds, folder in zip(self.datasets, folders):
            if len(ds) == 0:
                raise ValueError(f'No images found in folder "{folder}" with supported extensions.')
        self.dataloaders = [cycle(data.DataLoader(ds, batch_size=train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=8, drop_last=True)) for ds in self.datasets]
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0
        self.reset_parameters()
        self.fid_eval_every = save_and_sample_every

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

    def train(self):
        backwards = lambda loss: loss.backward()
        acc_loss = 0
        if self.use_wandb and self.wandb_run is None:
            import wandb
            self.wandb_run = wandb.init(project=self.wandb_project, name=self.wandb_run_name)
        while self.step < self.train_num_steps:
            u_loss = 0
            for _ in range(self.gradient_accumulate_every):
                x_target = next(self.dataloaders[0]).cuda()
                _model = self.model.module if hasattr(self.model, 'module') else self.model
                s = int(torch.randint(1, _model.num_timesteps + 1, (1,), device=x_target.device).item())
                xs_list = [x_target] + [next(self.dataloaders[1]).cuda() for _ in range(s)]
                step = torch.full((x_target.shape[0],), s - 1, dtype=torch.long, device=x_target.device)
                loss = torch.mean(self.model(xs_list, t=step))
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every)
            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)
            self.opt.step()
            self.opt.zero_grad()
            if self.use_wandb and self.wandb_run is not None:
                import wandb
                wandb.log({"train_loss": u_loss / self.gradient_accumulate_every, "step": self.step})
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size
                with torch.no_grad():
                    batch_list = [next(dl).cuda() for dl in self.dataloaders]
                    _ema = self.ema_model.module if hasattr(self.ema_model, 'module') else self.ema_model
                    x1_0s, x_ts = _ema.all_sample(batch_size=batches, xs=batch_list)
                    og_imgs = batch_list[0]
                    og_imgs = (og_imgs + 1) * 0.5
                    utils.save_image(og_imgs, str(self.results_folder / f'sample-og-{milestone}.png'), nrow=6)
                    recons = (x1_0s[0] + 1) * 0.5
                    utils.save_image(recons, str(self.results_folder / f'sample-recon-{milestone}.png'), nrow=6)
                    xt0 = (x_ts[0] + 1) * 0.5
                    utils.save_image(xt0, str(self.results_folder / f'sample-xt0-{milestone}.png'), nrow=6)
                    if self.use_wandb and self.wandb_run is not None:
                        import wandb
                        from torchvision.utils import make_grid
                        wandb.log({
                            "og": wandb.Image(make_grid(og_imgs, nrow=6)),
                            "recon": wandb.Image(make_grid(recons, nrow=6)),
                            "xt0": wandb.Image(make_grid(xt0, nrow=6)),
                            "milestone": milestone,
                            "step": self.step
                        })
                    # FID evaluation (original vs reconstructed)
                    try:
                        fid_value = calculate_fid_given_samples(
                            samples=[og_imgs.cpu(), ((x1_0s[-1] + 1) * 0.5).cpu()],
                            batch_size=min(50, og_imgs.shape[0]),
                            device='cuda:0' if torch.cuda.is_available() else 'cpu',
                            dims=2048,
                            num_workers=1
                        )
                        if self.use_wandb and self.wandb_run is not None:
                            import wandb
                            wandb.log({"fid_recon_vs_real": float(fid_value), "step": self.step})
                    except Exception as e:
                        pass
                self.save()
            self.step += 1
        if self.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.finish()

    @torch.no_grad()
    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        batch_list = [next(dl).cuda() for dl in self.dataloaders]
        _ema = self.ema_model.module if hasattr(self.ema_model, 'module') else self.ema_model
        X_0s, X_ts = _ema.all_sample(batch_size=batches, xs=batch_list, times=s_times)
        og_img = (batch_list[0] + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)
        import imageio
        frames_t = []
        frames_0 = []
        for i in range(len(X_0s)):
            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))
            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)
        if self.use_wandb and self.wandb_run is None:
            import wandb
            self.wandb_run = wandb.init(project=self.wandb_project, name=self.wandb_run_name)
        if self.use_wandb and self.wandb_run is not None:
            import wandb
            from torchvision.utils import make_grid
            wandb.log({
                "test_og": wandb.Image(make_grid(og_img, nrow=6)),
                "test_x0_last": wandb.Image(make_grid((X_0s[-1] + 1) * 0.5, nrow=6)),
                "test_xt_first": wandb.Image(make_grid((X_ts[0] + 1) * 0.5, nrow=6))
            })
        try:
            fid_value = calculate_fid_given_samples(
                samples=[og_img.cpu(), ((X_0s[-1] + 1) * 0.5).cpu()],
                batch_size=min(50, og_img.shape[0]),
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                dims=2048,
                num_workers=1
            )
            if self.use_wandb and self.wandb_run is not None:
                import wandb
                wandb.log({"fid_recon_vs_real_test": float(fid_value)})
        except Exception:
            pass
