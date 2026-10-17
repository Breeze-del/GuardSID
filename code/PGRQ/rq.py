import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_e_list, e_dim, sk_epsilons=None, kmeans_init=True,
                 kmeans_iters=100, sk_iters=50, use_linear=0, use_sk=False,
                 beta=0.25, diversity_loss=0.0, tau_vq=0.1):
        super().__init__()
        if use_sk or use_linear or diversity_loss:
            raise ValueError("Paper-aligned PGRQ uses softmax/ST assignments and a separate utilization penalty; "
                             "disable Sinkhorn, linear quantization and the old diversity loss")
        self.n_e_list = tuple(int(x) for x in n_e_list)
        self.e_dim = int(e_dim)
        self.num_quantizers = len(self.n_e_list)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim, tau_vq=tau_vq, beta=beta)
            for n_e in self.n_e_list
        ])

    @torch.no_grad()
    def initialize_codebooks(self, encoded, random_state=2024):
        """Sequential k-means on progressively quantized encoder residuals."""
        residual = encoded.detach().float().cpu().numpy()
        for quantizer in self.vq_layers:
            if len(residual) < quantizer.n_e:
                raise ValueError("Fewer training POIs than codewords")
            km = KMeans(n_clusters=quantizer.n_e,
                        n_init=1, max_iter=self.kmeans_iters,
                        random_state=random_state)
            centers = km.fit(residual).cluster_centers_.astype(np.float32)
            quantizer.embedding.weight.copy_(torch.tensor(centers,
                device=quantizer.embedding.weight.device,
                dtype=quantizer.embedding.weight.dtype))
            residual -= centers[km.predict(residual)]

    def get_codebook(self):
        # Return a list: codebooks can have different cardinalities.
        return [vq.get_codebook() for vq in self.vq_layers]

    def forward(self, x, epoch_idx=None, return_soft=False):
        residual = x
        quantized_total = torch.zeros_like(x)
        all_indices, all_distances, all_soft = [], [], []
        quant_loss_per_poi = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for quantizer in self.vq_layers:
            x_res, loss_per_poi, index, dist, soft = quantizer(residual, epoch_idx)
            quantized_total = quantized_total + x_res
            residual = residual - x_res
            quant_loss_per_poi = quant_loss_per_poi + loss_per_poi
            all_indices.append(index)
            all_distances.append(dist)
            all_soft.append(soft)
        result = (quantized_total, quant_loss_per_poi,
                  torch.stack(all_indices, -1), all_distances)
        return (*result, all_soft) if return_soft else result
