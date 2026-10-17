
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rq import ResidualVectorQuantizer
from .exposure import ConditionalPrefixLeakage


def make_mlp(widths, dropout=0., bn=False):
    blocks = []
    for start, stop in zip(widths[:-2], widths[1:-1]):
        blocks.append(nn.Linear(start, stop))
        if bn:
            blocks.append(nn.BatchNorm1d(stop))
        blocks.extend([nn.ReLU(), nn.Dropout(dropout)])
    blocks.append(nn.Linear(widths[-2], widths[-1]))
    return nn.Sequential(*blocks)


class PGRQ(nn.Module):
    def __init__(self, in_dim=16, num_emb_list=(32, 32, 32), e_dim=64,
                 layers=(512, 256, 128), dropout_prob=0.1, bn=False,
                 loss_type='mse', quant_loss_weight=0.1, kmeans_init=True,
                 kmeans_iters=100, sk_epsilons=None, sk_iters=50, use_sk=False,
                 use_linear=0, beta=0.25, diversity_loss=0.0,
                 tau_vq=0.1, protected_levels=2, mu=None, gamma=0.1,
                 utilization_target=0.5, shrinkage=1.0,
                 ref_smoothing=1e-6, prefix_chunk=512):
        super().__init__()
        if loss_type != 'mse':
            raise ValueError("Eq. (8)-(9) requires the squared L2 reconstruction loss")
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list)
        self.e_dim = e_dim
        self.mu = quant_loss_weight if mu is None else float(mu)
        self.gamma = float(gamma)
        self.nu = float(utilization_target)
        widths = [in_dim] + list(layers) + [e_dim]
        self.encoder = make_mlp(widths, dropout_prob, bn)
        self.decoder = make_mlp([e_dim] + list(layers) + [in_dim],
                                dropout_prob, bn)
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                     sk_epsilons=sk_epsilons, kmeans_init=kmeans_init,
                     kmeans_iters=kmeans_iters, sk_iters=sk_iters,
                     use_linear=use_linear, use_sk=use_sk, beta=beta,
                     diversity_loss=diversity_loss, tau_vq=tau_vq)
        self.guard = ConditionalPrefixLeakage(num_emb_list,
                     protected_levels=protected_levels, shrinkage=shrinkage,
                     ref_smoothing=ref_smoothing, prefix_chunk=prefix_chunk)
        # No output sigmoid: one-hot/multi-hot blocks coexist with signed
        # offline LightGCN features in the l2-normalized descriptor.

    def forward(self, xs, epoch_idx=0, return_details=False):
        x_e = self.encoder(xs)
        x_q, losses, indices, distances, softs = self.rq(
            x_e, epoch_idx, return_soft=True)
        recon = self.decoder(x_q)
        if return_details:
            return recon, losses, indices, softs
        return recon, losses.mean(), indices

    @torch.no_grad()
    def get_indices(self, xs, epoch_idx=0):
        encoded = self.encoder(xs)
        q, _, indices, distances = self.rq(encoded, epoch_idx)
        return q, indices, distances

    @torch.no_grad()
    def init_codebooks(self, all_features, seed=2024):
        was_training = self.training
        self.eval()
        encoded = self.encoder(all_features)
        self.rq.initialize_codebooks(encoded, random_state=seed)
        self.train(was_training)

    def compute_loss(self, out, quant_loss, xs=None):
        # Eq. (8): mean over POIs of squared l2 norm, NOT mean over all entries.
        recon = (out - xs).square().sum(-1).mean()
        return recon+self.mu*quant_loss, recon

    def guarded_loss(self, xs, buckets, context_sizes, eta, eps, epoch_idx=0):
        """POI-mass-weighted stochastic estimate over sampled whole contexts.

        Contexts were drawn with replacement with p(c)=N_c/|P|. Each context
        occupies a contiguous slice and includes ALL its POIs. Repeated
        context draws are intentionally kept as separate slices.
        """
        out, quant_per_poi, _, soft = self(xs, epoch_idx, return_details=True)
        if sum(context_sizes) != len(xs):
            raise ValueError("Context sizes do not match concatenated samples")
        error = (out-xs).square().sum(-1)
        recon_parts, quant_parts, util_parts, leak_parts = [], [], [], []
        at = 0
        for size in context_sizes:
            sl = slice(at, at+size)
            recon_parts.append(error[sl].mean())
            quant_parts.append(quant_per_poi[sl].mean())
            util_parts.append([a[sl].mean(0) for a in soft])
            leak_parts.append(self.guard.per_context(
                [a[sl] for a in soft], buckets[sl]))
            at += size
        recon = torch.stack(recon_parts).mean()
        quant = torch.stack(quant_parts).mean()
        utilization = 0.0
        for j, n_codes in enumerate(self.num_emb_list):
            u = torch.stack([c[j] for c in util_parts]).mean(0)
            utilization = utilization + F.relu(self.nu/n_codes-u).square().sum()
        leak = torch.stack(leak_parts).mean(0)  # one value per protected level
        over = F.relu(leak-eps)
        penalty = (eta.detach()*over).sum()
        total = recon+self.mu*quant+self.gamma*utilization+penalty
        return total, dict(recon=recon, quant=quant, util=utilization,
                           leak=leak, excess=over, penalty=penalty)
