import train_airplane as train
import os
import torch
import argparse
from torch.utils.data import RandomSampler
import logging
from dataset.dataset import AirplaneDataLoader, AirplaneDataset, load_or_compute_norm_stats
from models.Transolver_plus import Model
import torch.distributed as dist
import datetime
import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/airplane_data/')
parser.add_argument('--save_dir', default='/data/airplane_data/')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--cfd_model')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--nb_epochs', default=200, type=int)
parser.add_argument('--preprocessed', default=1, type=int)
parser.add_argument('--finetune', default=0, type=int)
# add arguments related to normalization
parser.add_argument('--pos_norm', default=1, type=int)
parser.add_argument('--out_norm', default=1, type=int)
parser.add_argument('--dataset', default='drivernet')
parser.add_argument('--eval', default=False, type=bool)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--out-dim', default=4, type=int)
args = parser.parse_args()
print(args)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
port = os.environ.get("MASTER_PORT", "64209")
hosts = int(os.environ.get("WORLD_SIZE", "1"))  # total number of processes
rank = int(os.environ.get("RANK", "0"))  # global process id
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
gpus = torch.cuda.device_count()  # gpus per node
args.local_rank = local_rank

# Only initialize a process group for true multi-process runs. On a single GPU we
# skip it entirely: the eidetic-state all_reduce is a no-op for world_size==1, and
# this avoids NCCL-not-built / libuv / gloo+CUDA issues (e.g. on Windows).
if hosts > 1:
    os.environ.setdefault("USE_LIBUV", "0")  # Windows torch is built without libuv
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend, init_method=f"tcp://{ip}:{port}", world_size=hosts,
                            rank=rank, timeout=datetime.timedelta(seconds=100))

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

train_dataset = AirplaneDataset(args.save_dir, train=True)
val_dataset = AirplaneDataset(args.save_dir, train=False)

train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(0))
val_sampler = RandomSampler(val_dataset, generator=torch.Generator().manual_seed(0))

train_loader = AirplaneDataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
val_loader = AirplaneDataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

# Normalization statistics are derived from the training set (cached to
# <save_dir>/norm_stats.json) so the pipeline migrates to new datasets without
# hand-edited constants.
stats = load_or_compute_norm_stats(args.save_dir, train_dataset.f_list)
pos_mean = stats['pos_mean'].cuda()
pos_std = stats['pos_std'].cuda()
norm_mean = stats['norm_mean'].cuda()
norm_std = stats['norm_std'].cuda()
out_mean = stats['out_mean'].cuda()
out_std = stats['out_std'].cuda()

model = Model(n_hidden=256, n_layers=4, space_dim=7,
                fun_dim=0,
                n_head=8,
                mlp_ratio=2, out_dim=6,
                slice_num=32,
                unified_pos=0,
                dropout=0.1).cuda()
# default
# path = f'metrics/airplane/{args.cfd_model}/{args.dataset}/{args.fold_id}/{args.nb_epochs}_{args.weight}'

# All training/eval artifacts (logs, checkpoints, json, predictions) go here.
path = "train"

if not os.path.exists(path):
    os.makedirs(path)

if args.eval:
    logging.basicConfig(filename=os.path.join(path, 'test.log'), level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    logging.info(args)
else:
    logging.basicConfig(filename=os.path.join(path, 'train.log'), level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    logging.info(args)

logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
logging.info(model)
print(model)

if not args.eval:
    # train
    model = train.main(device, train_loader, val_loader, model, hparams, path, val_iter=args.val_iter, reg=args.weight, pos_norm=args.pos_norm, out_norm=args.out_norm, norm_norm=0, pos_mean=pos_mean, pos_std=pos_std, out_mean=out_mean, out_std=out_std, norm_mean=norm_mean, norm_std=norm_std, full=True)
else:
    # Offline evaluation on the test split. Samples come from the test_set manifest
    # (val_dataset); Ma/alpha/beta and fields are read from each h5. result.csv, when
    # present, is used only to report the reference aerodynamic coefficients alongside.
    ckpt_file = os.path.join(path, f'model_{args.nb_epochs}.pth')
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(
            f"checkpoint {ckpt_file!r} not found - train first (omit --eval) to produce it")
    out_dir = os.path.join(path, 'output')
    os.makedirs(out_dir, exist_ok=True)

    res_file = os.path.join(args.save_dir, 'result.csv')
    coeff_df = pd.read_csv(res_file) if os.path.exists(res_file) else None

    model = torch.load(ckpt_file, weights_only=False).cuda()
    model.eval()
    l2re = 0.0
    n_samples = 0
    for h5_path in val_dataset.f_list:
        with h5py.File(h5_path, 'r') as f:
            pos = torch.from_numpy(f['pos'][:]).view(1, -1, 3).float().cuda()
            normals = torch.from_numpy(f['normals'][:]).view(1, -1, 3).float().cuda()
            values = torch.from_numpy(f['values'][:]).view(1, -1, 6).float().cuda()
            Ma, alpha, beta = float(f.attrs['Ma']), float(f.attrs['alpha']), float(f.attrs['beta'])

        with torch.no_grad():
            pos_n = (pos - pos_mean) / pos_std if args.pos_norm else pos
            N = pos_n.shape[1]
            x = torch.cat([pos_n, torch.zeros((1, N, 1), device=device), normals], dim=2)
            condition = torch.tensor([Ma, alpha, beta]).view(1, 3).float().cuda()
            out = model((x, pos_n, condition))
            if args.out_norm:
                out = out * out_std + out_mean
            sample_l2re = (torch.norm(out[0, :, -1] - values[0, :, -1])
                           / torch.norm(values[0, :, -1])).item()

        l2re += sample_l2re
        n_samples += 1
        name = os.path.splitext(os.path.basename(h5_path))[0]
        np.save(os.path.join(out_dir, f"{name}.npy"), out.cpu().numpy())

        ref = ""
        if coeff_df is not None:
            match = coeff_df[(coeff_df['idx'].astype(int) == int(name.split('_')[0]))
                             & (coeff_df['Ma'] == Ma) & (coeff_df['alpha'] == alpha)
                             & (coeff_df['beta'] == beta)]
            if len(match):
                r = match.iloc[0]
                ref = f" | ref CA={r['CA']:.4f} CN={r['CN']:.4f} Cm={r['Cm']:.4f}"
        msg = f"{name}: pressure L2RE={sample_l2re:.4f}{ref}"
        print(msg)
        logging.info(msg)

    avg = l2re / max(n_samples, 1)
    print(f"Average pressure L2RE over {n_samples} test samples: {avg:.6f}")
    logging.info(f"Average pressure L2RE over {n_samples} test samples: {avg:.6f}")

