import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, utils
from demixing_diffusion_pytorch import Unet, DecomposeDiffusion
import imageio

def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

def list_images(folder, exts=('jpg', 'jpeg', 'png')):
    allowed = {'.' + e.lower() for e in exts}
    return [p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in allowed]

def load_images(folder, image_size, device):
    paths = list_images(folder)
    transform = build_transform(image_size)
    xs = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        xs.append(t)
    return xs

def load_weights(diffusion, weights_path, device):
    data = torch.load(weights_path, map_location=device)
    state = data['ema'] if isinstance(data, dict) and 'ema' in data else (data['model'] if isinstance(data, dict) and 'model' in data else data)
    try:
        diffusion.load_state_dict(state)
    except Exception:
        adjusted = {}
        for k, v in state.items():
            nk = k
            if hasattr(diffusion, 'module'):
                if not nk.startswith('module.'):
                    nk = 'module.' + nk
            else:
                if nk.startswith('module.'):
                    nk = nk.replace('module.', '')
            adjusted[nk] = v
        diffusion.load_state_dict(adjusted)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, type=str)
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--time_steps', default=50, type=int)
    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--save_folder', default='./results_decompose_test', type=str)
    parser.add_argument('--remove_time_embed', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--gif_fps', default=10, type=int)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_folder, exist_ok=True)
    xs = load_images(args.img_dir, args.image_size, device)
    if len(xs) == 0:
        raise RuntimeError('No images found in img_dir')
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=not args.remove_time_embed,
        residual=args.residual
    ).to(device)
    diffusion = DecomposeDiffusion(
        model,
        image_size=args.image_size,
        channels=3,
        timesteps=args.time_steps,
        loss_type='l1',
        num_sources=len(xs)
    ).to(device)
    if torch.cuda.device_count() > 1:
        diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))
    load_weights(diffusion, args.weights, device)
    if hasattr(diffusion, 'module'):
        X1_0s, X_ts = diffusion.module.all_sample(batch_size=1, xs=xs, times=args.time_steps)
    else:
        X1_0s, X_ts = diffusion.all_sample(batch_size=1, xs=xs, times=args.time_steps)
    mix = X_ts[-1][0]
    recon = X1_0s[-1][0]
    utils.save_image((mix + 1) * 0.5, os.path.join(args.save_folder, 'mix.png'))
    utils.save_image((recon + 1) * 0.5, os.path.join(args.save_folder, 'recon.png'))
    frames_xt = []
    frames_x0 = []
    frames_noise = []
    for i in range(len(X_ts)):
        xt = (X_ts[i][0] + 1) * 0.5
        x0 = (X1_0s[i][0] + 1) * 0.5
        noise = torch.clamp(X_ts[i][0] - X1_0s[i][0], -1, 1)
        noise = (noise + 1) * 0.5
        p_xt = os.path.join(args.save_folder, f'step_{i:03d}_xt.png')
        p_x0 = os.path.join(args.save_folder, f'step_{i:03d}_x0.png')
        p_n = os.path.join(args.save_folder, f'step_{i:03d}_noise.png')
        utils.save_image(xt, p_xt)
        utils.save_image(x0, p_x0)
        utils.save_image(noise, p_n)
        frames_xt.append(imageio.imread(p_xt))
        frames_x0.append(imageio.imread(p_x0))
        frames_noise.append(imageio.imread(p_n))
    duration = 1.0 / max(1, args.gif_fps)
    imageio.mimsave(os.path.join(args.save_folder, 'gif_xt.gif'), frames_xt, duration=duration)
    imageio.mimsave(os.path.join(args.save_folder, 'gif_x0.gif'), frames_x0, duration=duration)
    imageio.mimsave(os.path.join(args.save_folder, 'gif_noise.gif'), frames_noise, duration=duration)

if __name__ == '__main__':
    main()
