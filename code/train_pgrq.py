import argparse
import logging
import random
import numpy as np
import torch
from PGRQ.pgrq import PGRQ
from POIdataset import EmbDataset
from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description='GuardSID stage I: PGRQ')
    p.add_argument('--data_mode', choices=['NYC', 'TKY', 'CA'], default='TKY')
    p.add_argument('--data_path', default=None,
                   help='Aligned NPZ produced by prepare_inputs.py; old poi_info.csv alone is insufficient')
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--batch_size', type=int, default=128,
                   help='Only controls steps_per_epoch when that option is not set')
    p.add_argument('--steps_per_epoch', type=int, default=0)
    p.add_argument('--eval_step', type=int, default=50)
    p.add_argument('--learner', choices=['Adam', 'AdamW'], default='AdamW')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--dropout_prob', type=float, default=0.1)
    p.add_argument('--bn', action='store_true')
    p.add_argument('--loss_type', default='mse')
    p.add_argument('--kmeans_init', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--kmeans_iters', type=int, default=100)
    p.add_argument('--use_sk', action='store_true', help='Unsupported for paper PGRQ')
    p.add_argument('--use_liner', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_emb_list', type=int, nargs='+', default=None)
    p.add_argument('--e_dim', type=int, default=64)
    p.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128])
    p.add_argument('--tau_vq', type=float, default=0.1)
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--mu', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--nu', type=float, default=0.5)
    p.add_argument('--protected_levels', type=int, default=2)
    p.add_argument('--mb_contexts', type=int, default=8)
    p.add_argument('--pgrq_warmup_epochs', type=int, default=5)
    p.add_argument('--calibration_permutations', type=int, default=20)
    p.add_argument('--alpha_eta', type=float, default=0.01)
    p.add_argument('--alpha_s', type=float, default=1.0)
    p.add_argument('--reference_smoothing', type=float, default=1e-6)
    p.add_argument('--prefix_chunk', type=int, default=512)
    p.add_argument('--index_batch_size', type=int, default=1024)
    p.add_argument('--ckpt_dir', type=str, default='save')
    args = p.parse_args()
    if args.data_path is None:
        args.data_path = f'/datasets/{args.data_mode}/pgrq_input.npz'
    if args.num_emb_list is None:
        k = 32 if args.data_mode == 'NYC' else 64
        args.num_emb_list = [k, k, k]
    if args.epochs <= args.pgrq_warmup_epochs:
        raise ValueError('epochs must exceed the 5-epoch PGRQ warm-up')
    if args.mb_contexts <= 0 or args.calibration_permutations < 2:
        raise ValueError('Invalid context sampling or permutation count')
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    data = EmbDataset(args.data_path)
    model = PGRQ(in_dim=data.x.shape[1], num_emb_list=args.num_emb_list,
                 e_dim=args.e_dim, layers=args.layers,
                 dropout_prob=args.dropout_prob, bn=args.bn,
                 loss_type=args.loss_type, quant_loss_weight=args.mu,
                 kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                 use_sk=args.use_sk, use_linear=args.use_liner,
                 beta=args.beta, tau_vq=args.tau_vq,
                 protected_levels=args.protected_levels, mu=args.mu,
                 gamma=args.gamma, utilization_target=args.nu,
                 shrinkage=args.alpha_s,
                 ref_smoothing=args.reference_smoothing,
                 prefix_chunk=args.prefix_chunk)
    best_recon, collision_rate = Trainer(args, model, data).fit()
    print(f'Best training reconstruction: {best_recon:.6f}')
    print(f'Semantic-code collision fraction before suffix: {collision_rate:.4%}')
    print(f'Final SID mappings: {args.ckpt_dir}/poi_sid.csv')
