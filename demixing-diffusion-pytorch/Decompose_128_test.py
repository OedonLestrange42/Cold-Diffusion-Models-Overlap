from demixing_diffusion_pytorch import Unet, DecomposeDiffusion, DecomposeTrainer
import argparse
import torch
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import os
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=100000, type=int)
parser.add_argument('--save_folder', default='./results_decompose_test', type=str)
parser.add_argument('--data_paths', nargs='+', default=None, type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--wandb_project', default='DecomposeDiffusion', type=str)
parser.add_argument('--wandb_run_name', default=None, type=str)
parser.add_argument('--test_type', default='train_data', type=str)
parser.add_argument('--image_dir', default=None, type=str)

args = parser.parse_args()

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = DecomposeDiffusion(
    model,
    image_size=128,
    channels=3,
    timesteps=args.time_steps,
    loss_type=args.loss_type,
    num_sources=(len(args.data_paths) if args.data_paths is not None else 1)
).cuda()

diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

if args.load_path is not None and os.path.isfile(args.load_path):
    sd = torch.load(args.load_path, map_location='cuda')
    state = sd['ema'] if 'ema' in sd else (sd['model'] if 'model' in sd else sd)
    try:
        diffusion.load_state_dict(state)
    except:
        fixed = {('module.' + k if not k.startswith('module.') else k): v for k, v in state.items()}
        diffusion.load_state_dict(fixed, strict=False)

def _transform(image_size):
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

def _list_images(folder):
    exts = {'.jpg', '.jpeg', '.png'}
    paths = [p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in exts]
    return paths

def _load_tensor(path, tfm):
    img = Image.open(path).convert('RGB')
    t = tfm(img)
    return t.unsqueeze(0).cuda()

def _save_sequence(X_0s, X_ts, save_folder, tag):
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    frames_t = []
    frames_0 = []
    frames_n = []
    for i in range(len(X_0s)):
        x0 = (X_0s[i] + 1) * 0.5
        utils.save_image(x0, str(Path(save_folder) / f'sample-{i}-{tag}-x0.png'), nrow=1)
        frames_0.append(imageio.imread(str(Path(save_folder) / f'sample-{i}-{tag}-x0.png')))
        xt = (X_ts[i] + 1) * 0.5
        utils.save_image(xt, str(Path(save_folder) / f'sample-{i}-{tag}-xt.png'), nrow=1)
        frames_t.append(imageio.imread(str(Path(save_folder) / f'sample-{i}-{tag}-xt.png')))
        noise_i = torch.clamp(X_ts[i] - X_0s[i], -1, 1)
        noise_v = (noise_i + 1) * 0.5
        utils.save_image(noise_v, str(Path(save_folder) / f'sample-{i}-{tag}-noise.png'), nrow=1)
        frames_n.append(imageio.imread(str(Path(save_folder) / f'sample-{i}-{tag}-noise.png')))
    imageio.mimsave(str(Path(save_folder) / f'Gif-{tag}-x0.gif'), frames_0)
    imageio.mimsave(str(Path(save_folder) / f'Gif-{tag}-xt.gif'), frames_t)
    imageio.mimsave(str(Path(save_folder) / f'Gif-{tag}-noise.gif'), frames_n)

if (args.image_dir is not None) or (args.test_type == 'image_dir'):
    paths = _list_images(args.image_dir)
    if len(paths) == 0:
        raise ValueError('No images found in --image_dir')
    tfm = _transform(128)
    xs = []
    for p in paths:
        xs.append(_load_tensor(p, tfm))
    _model = diffusion.module if hasattr(diffusion, 'module') else diffusion
    _model.num_sources = len(xs)
    X_0s, X_ts = _model.all_sample(batch_size=1, xs=xs, times=args.time_steps)
    tag = os.path.basename(os.path.abspath(args.image_dir))
    xt_last = (X_ts[-1] + 1) * 0.5
    utils.save_image(xt_last, str(Path(args.save_folder) / f'overlap-{tag}.png'), nrow=1)
    _save_sequence(X_0s, X_ts, args.save_folder, tag)
elif args.test_type in ['train_data', 'test_data']:
    if args.data_paths is None or len(args.data_paths) != 2:
        raise ValueError('Require --data_paths with exactly two folders: target first, interference second.')
    trainer = DecomposeTrainer(
        diffusion,
        args.data_paths,
        image_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=args.train_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        results_folder=args.save_folder,
        dataset=('train' if args.test_type == 'train_data' else 'test'),
        use_wandb=True,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    trainer.test_from_data(args.test_type.split('_')[0], s_times=None)
else:
    raise ValueError('Specify --image_dir for overlap test or use --data_paths with test_type.')
