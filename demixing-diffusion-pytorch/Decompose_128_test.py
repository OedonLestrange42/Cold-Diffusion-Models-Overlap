from demixing_diffusion_pytorch import Unet, DecomposeDiffusion, DecomposeTrainer
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=100000, type=int)
parser.add_argument('--save_folder', default='./results_decompose_test', type=str)
parser.add_argument('--data_paths', nargs='+', required=True, type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--wandb_project', default='DecomposeDiffusion', type=str)
parser.add_argument('--wandb_run_name', default=None, type=str)
parser.add_argument('--test_type', default='train_data', type=str)

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
    num_sources=len(args.data_paths)
).cuda()

diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

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
    dataset='train',
    use_wandb=True,
    wandb_project=args.wandb_project,
    wandb_run_name=args.wandb_run_name
)

if args.test_type == 'train_data':
    trainer.test_from_data('train', s_times=None)
elif args.test_type == 'test_data':
    trainer.test_from_data('test', s_times=None)
