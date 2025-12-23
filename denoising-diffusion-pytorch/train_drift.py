from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.drift_diffusion import DriftDiffusion, DriftTrainer
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int,
                    help="The number of steps the scheduler takes to go from clean image to an isotropic gaussian.")
parser.add_argument('--train_steps', default=700000, type=int,
                    help='The number of iterations for training.')
parser.add_argument('--save_folder', default='./results_drift', type=str)
parser.add_argument('--path1', type=str, required=True, help='Path to first image category')
parser.add_argument('--path2', type=str, required=True, help='Path to second image category')
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='default', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--batch_size', default=32, type=int)

args = parser.parse_args()
print(args)

# Create save folder if not exists
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = DriftDiffusion(
    model,
    image_size = 128,
    channels = 3,
    timesteps = args.time_steps,
    loss_type = args.loss_type,
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine,
    beta_schedule1='cosine',
    beta_schedule2='linear'
).cuda()

diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = DriftTrainer(
    diffusion,
    folder1=args.path1,
    folder2=args.path2,
    image_size = 128,
    train_batch_size = args.batch_size,
    train_lr = 2e-5,
    train_num_steps = args.train_steps,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    fp16 = False,
    results_folder = args.save_folder,
    load_path = args.load_path
)

trainer.train()
