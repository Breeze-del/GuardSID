import torch
from torch import nn
from torch.nn import functional as F


class PGRQ(nn.Module):
    def __init__(
        self,
        in_dim=16,
        num_emb_list=(32, 32, 32),
        e_dim=64,
        layers=(128, 64),
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=0.1,
        kmeans_init=True,
        kmeans_iters=20,
        sk_epsilons=None,
        sk_iters=20,
        use_sk=False,
        use_linear=0,
        beta=0.25,
        diversity_loss=0.0,
        protected_levels=2,
        tau_vq=0.1,
        util_loss_weight=0.1,
        min_usage=0.5,
        shrinkage=1.0,
        reference_smoothing=1e-6,
        dual_lr=0.01,
    ):
        super().__init__()

        # 保留旧参数入口，但禁止混入不属于本文目标的量化策略。
        if use_sk or use_linear or diversity_loss != 0:
            raise ValueError(
                "PGRQ uses nearest-code RQ and Lutil; "
                "disable SK/linear/diversity."
            )
        if sk_epsilons is not None and any(v != 0 for v in sk_epsilons):
            raise ValueError("Nonzero Sinkhorn epsilons are unsupported.")
        if loss_type != "mse":
            raise ValueError(
                "The manuscript uses squared-error reconstruction."
            )

        self.num_emb_list = tuple(num_emb_list)
        self.Lg = int(protected_levels)

        if not self.num_emb_list or min(self.num_emb_list) < 1:
            raise ValueError("Codebook sizes must be positive.")
        if not 1 <= self.Lg <= len(self.num_emb_list):
            raise ValueError("Invalid protected_levels.")
        if min(tau_vq, shrinkage, reference_smoothing) <= 0:
            raise ValueError("Temperature and smoothing must be positive.")
        if min(
            quant_loss_weight, util_loss_weight, beta, dual_lr
        ) < 0:
            raise ValueError(
                "Loss weights and dual_lr must be nonnegative."
            )
        if not 0 <= min_usage <= 1 or kmeans_iters < 1:
            raise ValueError("Invalid min_usage or kmeans_iters.")

        self.in_dim = in_dim
        self.beta = beta
        self.tau_vq = tau_vq
        self.mu = quant_loss_weight
        self.gamma = util_loss_weight
        self.nu = min_usage
        self.alpha_s = shrinkage
        self.ref_smoothing = reference_smoothing
        self.dual_lr = dual_lr
        self.kmeans_iters = kmeans_iters

        self.encoder = self._mlp(
            [in_dim, *layers, e_dim], dropout_prob, bn
        )
        self.decoder = self._mlp(
            [e_dim, *reversed(layers), in_dim], dropout_prob, bn
        )

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(k, e_dim))
            for k in self.num_emb_list
        ])
        for codebook in self.codebooks:
            nn.init.xavier_uniform_(codebook)

        # epsilon 和 eta 不参与 optimizer 更新。
        self.register_buffer(
            "initialized", torch.tensor(not kmeans_init)
        )
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("epsilon", torch.zeros(self.Lg))
        self.register_buffer("eta", torch.zeros(self.Lg))

        # 下列统计量只使用训练集构造。
        self.register_buffer(
            "context_index", torch.empty(0, dtype=torch.long)
        )
        self.register_buffer(
            "buckets", torch.empty(0, 4, dtype=torch.long)
        )
        self.register_buffer(
            "context_sizes", torch.empty(0, dtype=torch.long)
        )
        self.register_buffer(
            "context_reference", torch.empty(0, 4, 5)
        )

    @staticmethod
    def _mlp(dims, dropout, bn):
        modules = []
        for i, (din, dout) in enumerate(
            zip(dims[:-1], dims[1:])
        ):
            linear = nn.Linear(din, dout)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.append(linear)

            if i < len(dims) - 2:
                if bn:
                    modules.append(nn.BatchNorm1d(dout))
                modules.append(nn.ReLU())
                if dropout:
                    modules.append(nn.Dropout(dropout))

        # 最后一层不加 sigmoid，允许重构含负值的协同特征。
        return nn.Sequential(*modules)

    @staticmethod
    def _distance(x, codebook):
        return (
            x.square().sum(1, keepdim=True)
            + codebook.square().sum(1).unsqueeze(0)
            - 2 * x @ codebook.T
        ).clamp_min(0)

    @torch.no_grad()
    def set_exposure_statistics(
        self, context_ids, exposure_buckets
    ):
        """
        context_ids: [N]
            所有训练 POI 的最终上下文标签。
            调用前完成论文中的层级回退和小组归并。

        exposure_buckets: [N, 4]
            上下文内中心化、全体训练 POI 分位数离散化后的桶。
            四列对应 popularity / geo / time / user。
            每列取值为 0..4。

        两个输入的行顺序必须与 all_x 一致。
        """
        device = self.eta.device
        context_ids = torch.as_tensor(
            context_ids, dtype=torch.long, device=device
        )
        buckets = torch.as_tensor(
            exposure_buckets, dtype=torch.long, device=device
        )

        if context_ids.ndim != 1 or context_ids.numel() == 0:
            raise ValueError(
                "context_ids must have shape [number_of_POIs]."
            )
        if buckets.shape != (context_ids.numel(), 4):
            raise ValueError(
                "exposure_buckets must have shape "
                "[number_of_POIs, 4]."
            )
        if (buckets < 0).any() or (buckets >= 5).any():
            raise ValueError(
                "Convert exposure buckets to integer labels 0..4."
            )

        _, context_index = torch.unique(
            context_ids, sorted=True, return_inverse=True
        )
        sizes = torch.bincount(context_index)

        if (sizes < 10).any():
            raise ValueError(
                "Apply the manuscript's context backoff/merge first; "
                "each group needs >=10 POIs."
            )

        counts = torch.zeros(
            sizes.numel(), 4, 5, device=device
        )
        counts.index_add_(
            0, context_index, F.one_hot(buckets, 5).float()
        )

        self.context_index = context_index
        self.context_sizes = sizes
        self.buckets = buckets.clone()

        # 固定的上下文参考分布 pi_c。
        self.context_reference = (
            (counts + self.ref_smoothing)
            / (
                sizes[:, None, None]
                + 5 * self.ref_smoothing
            )
        )

        self.calibrated.fill_(False)
        self.epsilon.zero_()
        self.eta.zero_()

    @torch.no_grad()
    def initialize_codebooks(self, all_x):
        """使用训练描述符的编码结果，逐层执行残差 k-means。"""
        if bool(self.initialized):
            return

        was_training = self.training
        self.eval()

        try:
            residual = self.encoder(all_x).float()
            if residual.shape[0] == 0:
                raise ValueError("Empty initialization data.")

            for codebook in self.codebooks:
                n = residual.shape[0]
                k = codebook.shape[0]

                if n >= k:
                    ids = torch.randperm(
                        n, device=residual.device
                    )[:k]
                else:
                    ids = torch.randint(
                        n, (k,), device=residual.device
                    )

                centers = residual[ids].clone()

                for _ in range(self.kmeans_iters):
                    ids = self._distance(
                        residual, centers
                    ).argmin(1)
                    counts = torch.bincount(
                        ids, minlength=k
                    )
                    sums = torch.zeros_like(centers)
                    sums.index_add_(0, ids, residual)

                    nonempty = counts > 0
                    centers[nonempty] = (
                        sums[nonempty]
                        / counts[nonempty, None]
                    )

                codebook.copy_(centers)
                ids = self._distance(
                    residual, centers
                ).argmin(1)
                residual = residual - centers[ids]

            self.initialized.fill_(True)
        finally:
            self.train(was_training)

    def _quantize(self, z):
        if not bool(self.initialized):
            raise RuntimeError(
                "Call initialize_codebooks("
                "all_training_descriptors) first."
            )

        residual = z.float()
        quantized = torch.zeros_like(residual)
        quant_loss = residual.new_zeros(residual.shape[0])
        indices, assignments, distances = [], [], []

        for codebook in self.codebooks:
            codebook = codebook.float()
            distance = self._distance(residual, codebook)

            soft = F.softmax(
                -distance / self.tau_vq, dim=-1
            )
            index = distance.argmin(dim=-1)
            hard = F.one_hot(
                index, codebook.shape[0]
            ).to(soft.dtype)
            selected = F.embedding(index, codebook)

            # Eq. (8)：特征维求和，暂时保留逐 POI 损失。
            quant_loss = quant_loss + (
                (residual.detach() - selected)
                .square().sum(-1)
                + self.beta
                * (residual - selected.detach())
                .square().sum(-1)
            )

            # 硬前向，软分配反向。
            straight_through = hard + (
                soft - soft.detach()
            )
            selected_st = straight_through @ codebook

            quantized = quantized + selected_st
            residual = residual - selected_st

            indices.append(index)
            assignments.append(soft)
            distances.append(distance)

        info = {
            "quant_per_poi": quant_loss,
            "soft": assignments,
            "distances": distances,
        }
        return (
            quantized,
            info,
            torch.stack(indices, dim=1),
        )

    def forward(self, x, epoch_idx=None):
        """
        epoch_idx 保留为兼容参数。
        是否启用曝光惩罚由 calibrated 状态控制。
        """
        quantized, info, indices = self._quantize(
            self.encoder(x)
        )
        return self.decoder(quantized), info, indices

    @torch.no_grad()
    def get_indices(self, xs, epoch_idx=0):
        was_training = self.training
        self.eval()
        try:
            quantized, info, indices = self._quantize(
                self.encoder(xs)
            )
            return quantized, indices, info["distances"]
        finally:
            self.train(was_training)

    @torch.no_grad()
    def sample_context_batch(self, num_contexts=8):
        """
        按 N_c / N 有放回抽取上下文，并包含其全部 POI。

        draw_ids 区分每次抽样：
        同一个上下文被抽到两次时，不合并这两次抽样。
        """
        if (
            self.context_sizes.numel() == 0
            or num_contexts < 1
        ):
            raise ValueError(
                "Set exposure statistics and use "
                "num_contexts >= 1."
            )

        sampled = torch.multinomial(
            self.context_sizes.float(),
            num_contexts,
            replacement=True,
        )

        poi_ids, draw_ids = [], []
        for draw, context in enumerate(sampled):
            ids = torch.where(
                self.context_index == context
            )[0]
            poi_ids.append(ids)
            draw_ids.append(
                torch.full_like(ids, draw)
            )

        return torch.cat(poi_ids), torch.cat(draw_ids)

    def _layout(self, poi_ids, draw_ids):
        if self.context_index.numel() == 0:
            raise RuntimeError(
                "Call set_exposure_statistics first."
            )

        ids = torch.as_tensor(
            poi_ids,
            dtype=torch.long,
            device=self.eta.device,
        )
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError(
                "poi_ids must be nonempty and one-dimensional."
            )
        if (
            (ids < 0).any()
            or (ids >= self.context_index.numel()).any()
        ):
            raise ValueError("poi_ids are out of range.")

        contexts = self.context_index[ids]
        full_batch = draw_ids is None

        if full_batch:
            # 无 draw_ids 时，只允许完整训练 POI 集。
            if (
                ids.numel() != self.context_index.numel()
                or ids.unique().numel() != ids.numel()
            ):
                raise ValueError(
                    "Without draw_ids, provide every "
                    "training POI exactly once."
                )
            groups = contexts
        else:
            groups = torch.as_tensor(
                draw_ids,
                dtype=torch.long,
                device=ids.device,
            )
            if groups.shape != ids.shape:
                raise ValueError(
                    "draw_ids and poi_ids must have "
                    "identical shapes."
                )

        group_keys = groups.unique()
        weights = torch.zeros(
            ids.numel(), device=ids.device
        )
        layout = []

        for key in group_keys:
            rows = torch.where(groups == key)[0]
            context = contexts[rows[0]]

            if not torch.all(contexts[rows] == context):
                raise ValueError(
                    "One draw cannot contain multiple contexts."
                )

            if (
                rows.numel()
                != int(self.context_sizes[context])
                or ids[rows].unique().numel()
                != rows.numel()
            ):
                raise ValueError(
                    "Each draw must contain every POI "
                    "of its context exactly once."
                )

            if full_batch:
                mass = rows.numel() / ids.numel()
            else:
                mass = 1.0 / group_keys.numel()

            weights[rows] = mass / rows.numel()
            layout.append((rows, context, mass))

        return ids, weights, layout

    @staticmethod
    def _js(p, q):
        midpoint = 0.5 * (p + q)
        tiny = torch.finfo(p.dtype).tiny

        return 0.5 * (
            p * (
                p.clamp_min(tiny).log()
                - midpoint.clamp_min(tiny).log()
            )
            + q * (
                q.clamp_min(tiny).log()
                - midpoint.clamp_min(tiny).log()
            )
        ).sum(-1)

    def _leakage(self, soft, buckets, layout):
        """Eq. (6)-(7)：逐上下文、逐前缀层计算曝光依赖。"""
        values = [
            soft[0].new_zeros(())
            for _ in range(self.Lg)
        ]

        for rows, context, context_mass in layout:
            reference = self.context_reference[context]

            labels = (
                F.one_hot(buckets[rows], 5)
                .float()
                .reshape(rows.numel(), 20)
            )

            prefix = soft[0][rows]

            for level in range(self.Lg):
                if level > 0:
                    # w_{p,g} = 各层软分配概率的乘积。
                    prefix = (
                        prefix.unsqueeze(-1)
                        * soft[level][rows].unsqueeze(1)
                    ).flatten(1)

                support = prefix.sum(0)
                counts = (
                    prefix.T @ labels
                ).reshape(-1, 4, 5)

                distribution = (
                    counts
                    + self.alpha_s * reference.unsqueeze(0)
                ) / (
                    support[:, None, None]
                    + self.alpha_s
                )

                # 四个代理分别计算 JSD，再取平均。
                divergence = self._js(
                    distribution,
                    reference.unsqueeze(0),
                ).mean(-1)

                value = (
                    context_mass
                    * (support * divergence).sum()
                    / rows.numel()
                )
                values[level] = values[level] + value

        return torch.stack(values)

    def compute_loss(
        self,
        out,
        rq_info,
        xs,
        poi_ids,
        draw_ids=None,
        return_metrics=False,
    ):
        """
        out、xs、poi_ids 必须具有相同的批次行顺序。

        draw_ids=None:
            完整训练 POI 集，按每个 POI 等权计算。

        draw_ids 非空:
            sample_context_batch 返回的上下文抽样批次，
            先在每次抽样内部平均，再在抽样之间平均。
        """
        ids, weights, layout = self._layout(
            poi_ids, draw_ids
        )

        if (
            out.shape != xs.shape
            or out.ndim != 2
            or out.shape[0] != ids.numel()
        ):
            raise ValueError(
                "out, xs and poi_ids must describe "
                "the same batch in the same order."
            )

        reconstruction = (
            (out.float() - xs.float())
            .square().sum(-1)
            * weights
        ).sum()

        quantization = (
            rq_info["quant_per_poi"] * weights
        ).sum()

        # 先估计全局码字使用频率，再应用下限惩罚。
        utilization = reconstruction.new_zeros(())
        for assignment, k in zip(
            rq_info["soft"], self.num_emb_list
        ):
            frequency = (
                weights[:, None] * assignment
            ).sum(0)

            utilization = utilization + F.relu(
                self.nu / k - frequency
            ).square().sum()

        leakage = self._leakage(
            rq_info["soft"],
            self.buckets[ids],
            layout,
        )

        penalty = reconstruction.new_zeros(())
        if bool(self.calibrated):
            penalty = (
                self.eta.detach().clone()
                * F.relu(leakage - self.epsilon)
            ).sum()

        total = (
            reconstruction
            + self.mu * quantization
            + self.gamma * utilization
            + penalty
        )

        if not return_metrics:
            return total, reconstruction

        metrics = {
            "recon": reconstruction.detach(),
            "quant": quantization.detach(),
            "util": utilization.detach(),
            "leakage": leakage.detach(),
            "exposure_penalty": penalty.detach(),
        }
        return total, reconstruction, metrics

    @torch.no_grad()
    def calibrate_tolerances(
        self,
        all_x,
        num_permutations=20,
        num_contexts=8,
    ):
        """
        在 warm-up 后调用一次：
        epsilon_l = 置换泄漏均值 + 一个标准差。

        使用与训练相同的上下文抽样、平滑和权重。
        """
        if (
            num_permutations < 2
            or all_x.shape[0]
            != self.context_index.numel()
        ):
            raise ValueError(
                "Use >=2 permutations and all training "
                "POIs in registered order."
            )

        was_training = self.training
        self.eval()

        try:
            measurements = []

            for _ in range(num_permutations):
                ids, draws = self.sample_context_batch(
                    num_contexts
                )
                _, _, layout = self._layout(ids, draws)

                # 置换整个四维曝光向量，保留代理间关系。
                # 相同上下文的重复抽样共享同一次置换。
                permuted = self.buckets.clone()

                for context in self.context_index[ids].unique():
                    rows = torch.where(
                        self.context_index == context
                    )[0]
                    order = torch.randperm(
                        rows.numel(), device=rows.device
                    )
                    permuted[rows] = self.buckets[
                        rows[order]
                    ]

                _, info, _ = self(all_x[ids])
                measurements.append(
                    self._leakage(
                        info["soft"],
                        permuted[ids],
                        layout,
                    )
                )

            values = torch.stack(measurements)
            self.epsilon.copy_(
                values.mean(0)
                + values.std(0, unbiased=True)
            )
            self.eta.zero_()
            self.calibrated.fill_(True)

            return self.epsilon.clone()
        finally:
            self.train(was_training)

    @torch.no_grad()
    def update_dual(self, leakage):
        """
        在 backward 和 optimizer.step 之后调用。

        eta_l <- max(
            0,
            eta_l + dual_lr * (leakage_l - epsilon_l)
        )
        """
        if not bool(self.calibrated):
            return

        leakage = leakage.detach().to(self.eta)
        if (
            leakage.shape != self.eta.shape
            or not torch.isfinite(leakage).all()
        ):
            raise ValueError("Invalid leakage for dual update.")

        self.eta.add_(
            self.dual_lr * (leakage - self.epsilon)
        ).clamp_(min=0)
