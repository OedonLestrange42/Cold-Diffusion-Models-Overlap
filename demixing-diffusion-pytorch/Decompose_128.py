from demixing_diffusion_pytorch import Unet
from demixing_diffusion_pytorch import SICDiffusion, SICTrainer
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--k_steps', default=5, type=int, help='Total number of images to superimpose (K)')
parser.add_argument('--train_steps', default=100000, type=int)
parser.add_argument('--save_folder', default='./results_sic', type=str)
parser.add_argument('--data_path', required=True, type=str, help='Path to dataset folder')
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--wandb_project', default='SIC_Diffusion', type=str)
parser.add_argument('--wandb_run_name', default=None, type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=2e-5, type=float)

args = parser.parse_args()

# Model
# Channels = 4 (3 for image + 1 for alpha weight map)
# Out dim = 3 (Predicted interference image)
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=4, 
    out_dim=3,
    with_time_emb=True
).cuda()

diffusion = SICDiffusion(
    model,
    image_size=128,
    channels=3,
    k_steps=args.k_steps,
    loss_type=args.loss_type
).cuda()

# DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    diffusion = torch.nn.DataParallel(diffusion)

trainer = SICTrainer(
    diffusion,
    args.data_path,
    image_size=128,
    train_batch_size=args.batch_size,
    train_lr=args.lr,
    train_num_steps=args.train_steps,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    fp16=False,
    results_folder=args.save_folder,
    wandb_project=args.wandb_project,
    wandb_run_name=args.wandb_run_name,
    load_path=args.load_path
)

trainer.train()
