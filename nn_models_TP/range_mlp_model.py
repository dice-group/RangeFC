import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ResBlock(nn.Module):
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1  = nn.Linear(d, 4*d)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(4*d, d)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        z = self.norm(x)
        z = self.fc2(self.act(self.fc1(z)))
        return x + self.drop(z)

class RangeMLPModel(pl.LightningModule):
    def __init__(self,
                 num_entities,
                 num_relations,
                 num_times,
                 embedding_dim=100,
                 hidden_dim=256,
                 dropout=0.3,
                 lr=1e-3,
                 t_dim=128,
                 num_experts=4,
                 k_experts=2,
                 gate_temp_start=1.6,
                 gate_temp_end=0.8,
                 gate_balance=0.03,
                 weight_decay=0.0,
                 max_num_epochs=80,
                 idx_time_dict=None,
                 order_penalty_lambda=0.0,
                 use_interaction=False,
                 loss_type="l1",
                 huber_beta=0.5,
                 use_prod=False,
                 end_weight=1.0,
                 extra_order_pen=0.0,
                 emb_noise=0.0,
                 w_ce=0.08,
                 w_iou=0.20,
                 w_consistency=0.01,
                 w_end_boost=0.02,
                 w_order=0.01,
                 w_align=0.01,
                 w_smooth=0.010,
                 w_ridge=0.0003,
                 w_reg=1.0,
                 pred_head="calendar",
                 ):
        super().__init__()
        self.name = 'RangeMLP'
        self.save_hyperparameters()
        # --- canonical time map: always expose idx -> year (int) ---
        self.idx_time_dict = idx_time_dict or {}

        self.idx_to_year = None  # list[int] length = num_times
        self.year_idx_dict = {}  # dict[int idx] -> int year (same info as idx_to_year)

        def _is_int_like(x):
            try:
                int(str(x));
                return True
            except Exception:
                return False

        # Build a dense array idx_to_year using num_times
        T = int(self.hparams.num_times)
        arr = [None] * T

        if self.idx_time_dict:
            # self.idx_time_dict at this point is year->idx (per your file)
            for year, idx in self.idx_time_dict.items():
                if not (_is_int_like(year) and _is_int_like(idx)):
                    continue
                i = int(str(idx))
                if 0 <= i < T:
                    arr[i] = int(str(year))

        # Fill any holes by nearest-neighbour so every 0..T-1 is defined
        last = None
        for i in range(T):
            if arr[i] is None:
                nxt = None
                for j in range(i + 1, T):
                    if arr[j] is not None:
                        nxt = arr[j]
                        break
                arr[i] = last if nxt is None else (last if last is not None else nxt)
            last = arr[i]

        self.idx_to_year = arr
        self.year_idx_dict = {i: y for i, y in enumerate(arr)}

        print(f"[DEBUG] idx->year sample: {self.idx_to_year[:5]}  (num_times={T})")

        # --- year normalization helpers ---
        # Build a numeric list of YEARS from idx->year dict
        try:
            _years = [int(float(y)) for y in self.year_idx_dict.values()]
        except Exception:
            _years = []
        if len(_years) == 0:
            _years = [0, 1]  # fallback if empty

        min_year = min(_years)
        max_year = max(_years)
        year_span = max(1, max_year - min_year)

        # keep as buffers so they move to device automatically
        self.register_buffer("min_year", torch.tensor(float(min_year)))
        self.register_buffer("max_year", torch.tensor(float(max_year)))
        self.register_buffer("year_span", torch.tensor(float(year_span)))

        def _year_to_norm_fn(y: torch.Tensor) -> torch.Tensor:
            # y in YEARS (float) → [0,1]
            return (y - self.min_year) / self.year_span

        def _norm_to_year_fn(n: torch.Tensor) -> torch.Tensor:
            # n in [0,1] → YEARS (float)
            return self.min_year + n * self.year_span

        # bind as methods
        self._year_to_norm = _year_to_norm_fn
        self._norm_to_year = _norm_to_year_fn

        self.embedding_dim = embedding_dim

       # self.ent_emb = nn.Embedding(num_entities, embedding_dim)
       # self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # --- unfreeze plan ---
        self.unfreeze_epoch = -1
        self._did_freeze_once = False

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times

        # in __init__, after you know self.num_relations and self.num_times
        self.register_buffer("rel_min_idx", torch.zeros(self.num_relations))
        self.register_buffer("rel_max_idx", torch.full((self.num_relations,),
                                                       float(self.num_times - 1)))
        self._ranges_frozen = False  # we freeze after a few epochs to avoid drift

        # --- relation-specific observed bands (learned from TRAIN only, then frozen) ---
        self.register_buffer("rel_min_obs", torch.full((self.num_relations,), float(self.num_times - 1)))
        self.register_buffer("rel_max_obs", torch.zeros(self.num_relations))
        self.band_freeze_epoch = getattr(self, "band_freeze_epoch", 3)  # freeze after N epochs
        self.band_margin = getattr(self, "band_margin", 1.5)  # widen the observed band a bit
        self.prior_warmup_epoch = 5

        # small learnable per-relation offsets (initialized 0)
        self.rel_bias_start = nn.Parameter(torch.zeros(self.num_relations))
        self.rel_bias_end = nn.Parameter(torch.zeros(self.num_relations))
        self.rel_bias_l2 = 1e-4  # regularization weight (fixed, no CLI)

        # --- duration prior buffers (learned online via EMA) ---
        self.register_buffer("rel_mu", torch.zeros(self.num_relations))
        self.register_buffer("rel_sigma", torch.ones(self.num_relations))
        self.register_buffer("rel_count", torch.zeros(self.num_relations))  # for debug/inspection
        self.prior_momentum = 0.05  # EMA step (try 0.02–0.1)
        self.prior_weight = 0.0  # loss weight (try 0.02–0.1)
        self._eps = 1e-6

        # ablation flags (off for now; we’ll re-enable after MoE is stable)
        self.use_bands = False  # disable banding this run
        self.use_prior = False  # disable prior loss this run


        self.lr = lr
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_num_epochs)
        # ---- model width & dropout (single change) ----
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)

        # --- SWA knobs ---
        self.use_swa = True  # flip to False to disable easily
        self.swa_start_epoch = max(5, int(0.9 * self.max_epochs))  # start near the end
        self.swa_update_freq = 1  # update every epoch
        self._swa_inited = False
        self._swa_model = None

        print(f"[DEBUG] RangeMLPModel: using lr={self.lr}")
        self.order_penalty_lambda = order_penalty_lambda
        self.use_interaction = use_interaction
        self.use_prod = use_prod
        self.end_weight = float(end_weight)
        self.extra_order_pen = float(extra_order_pen)
        self.emb_noise = float(emb_noise)  # 0.0 disables noise
        self.loss_type = loss_type
        self.huber_beta = huber_beta

        # Ablation weights
        self.w_ce = float(w_ce)
        self.w_iou = float(w_iou)
        self.w_consistency = float(w_consistency)
        self.w_end_boost = float(w_end_boost)
        self.w_order = float(w_order)
        self.w_align = float(w_align)
        self.w_smooth = float(w_smooth)
        self.w_ridge = float(w_ridge)
        self.w_reg = float(w_reg)
        self.pred_head = str(pred_head).lower()
        if self.pred_head not in {"calendar", "regression"}:
            raise ValueError(f"Unknown pred_head: {self.pred_head}")

        # Precompute scale for normalization <-> index space
        # idx in [0, num_times-1]  <->  norm in [0,1]
        self.register_buffer("_timescale", torch.tensor(float(max(1, num_times - 1))))

        # Embeddings (will be set from externally loaded tensors)
        self.ent_emb = nn.Embedding(num_entities, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # MLP layers
        #self.fc1 = nn.Linear(embedding_dim * 3, 512)
        # ---- wider trunk + 2-dim head ----
        parts = 3  # [h, r, t]
        if self.use_interaction:
            parts += 1  # |h - t|
        if self.use_prod:
            parts += 1  # h ⊙ r
        in_feats = embedding_dim * parts
        H = int(getattr(self, "hidden_dim", 1024))  # use the wider width you set above

        # trunk: 3x[Linear->GELU->Dropout]
        self.trunk = nn.Sequential(
            nn.Linear(in_feats, H),
            nn.GELU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

        self.t_dim = int(t_dim)
        self.time_emb = nn.Embedding(self.num_times, self.t_dim)  # shared calendar E

        # === Mixture-of-Experts (shared trunk; instance-aware gate) ===
        self.num_experts = int(num_experts)
        self.k_experts = int(min(k_experts, self.num_experts))
        self.gate_temp_start = float(gate_temp_start)
        self.gate_temp_end = float(gate_temp_end)
        self.gate_temp = float(gate_temp_start)  # initial temp
        self.gate_balance = float(gate_balance)

        H = int(self.hidden_dim)

        # experts: projections for time-heads + small 2-d regression head (shared trunk input 'z')
        self.exp_proj_s = nn.ModuleList([nn.Linear(H, self.t_dim) for _ in range(self.num_experts)])
        self.exp_proj_e = nn.ModuleList([nn.Linear(H, self.t_dim) for _ in range(self.num_experts)])
        self.exp_out2 = nn.ModuleList([nn.Linear(H, 2) for _ in range(self.num_experts)])

        # gate uses relation + a tiny projection of z (instance-aware)
        self.rel_gate_emb = nn.Embedding(self.num_relations, 32)
        self.gate_feat = nn.Sequential(nn.Linear(H, 32), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, self.num_experts))

        # inits
        nn.init.xavier_uniform_(self.time_emb.weight)
        for i in range(self.num_experts):
            nn.init.xavier_uniform_(self.exp_proj_s[i].weight);
            nn.init.zeros_(self.exp_proj_s[i].bias)
            nn.init.xavier_uniform_(self.exp_proj_e[i].weight);
            nn.init.zeros_(self.exp_proj_e[i].bias)
            nn.init.xavier_uniform_(self.exp_out2[i].weight);
            nn.init.zeros_(self.exp_out2[i].bias)

        for m in list(self.gate_feat.modules()) + list(self.gate.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

        # init: Xavier for trunk layers
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        #self.dropout = nn.Dropout(dropout)

        if loss_type.lower() == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type.lower() == "huber":
            self.loss_fn = nn.SmoothL1Loss(beta=huber_beta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Use Huber loss (SmoothL1Loss) for robustness and smoother convergence
        #self.loss_fn = nn.SmoothL1Loss(beta=0.5)
        #self.loss_fn = nn.L1Loss()

        # ---- helpers ----
    def _to_norm(self, idx_float: torch.Tensor) -> torch.Tensor:
        """Index -> normalized [0,1]."""
        return idx_float / self._timescale

    def _to_index(self, norm_float: torch.Tensor) -> torch.Tensor:
        """Normalized [0,1] -> index space (float)."""
        return norm_float * self._timescale

    def _soft_index_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)  # [B, T]
        idxs = torch.arange(self.num_times, device=logits.device, dtype=torch.float)  # [T]
        return (probs * idxs).sum(dim=-1)  # [B]

    def _giou_1d(self, ps_idx, pe_idx, y1_idx, y2_idx):
        # all tensors float [B], in index space
        # enforce order
        s_pred = torch.minimum(ps_idx, pe_idx)
        e_pred = torch.maximum(ps_idx, pe_idx)
        s_true = torch.minimum(y1_idx, y2_idx)
        e_true = torch.maximum(y1_idx, y2_idx)

        # lengths
        inter_left = torch.maximum(s_pred, s_true)
        inter_right = torch.minimum(e_pred, e_true)
        inter = torch.clamp(inter_right - inter_left + 1.0, min=0.0)

        union_left = torch.minimum(s_pred, s_true)
        union_right = torch.maximum(e_pred, e_true)
        union = (union_right - union_left + 1.0).clamp(min=1.0)

        iou = inter / union

        # generalized IoU: add hull penalty for non-overlap
        hull_left = union_left
        hull_right = union_right
        hull = (hull_right - hull_left + 1.0).clamp(min=1.0)
        giou = iou - (hull - union) / hull  # ∈ (-1,1]; higher is better

        loss = 1.0 - giou  # turn into loss
        return loss.mean()

    def _aeiou_like(self, ps_idx, pe_idx, y1_idx, y2_idx):
        # midpoints & lengths (float)
        s_pred = torch.minimum(ps_idx, pe_idx);
        e_pred = torch.maximum(ps_idx, pe_idx)
        s_true = torch.minimum(y1_idx, y2_idx);
        e_true = torch.maximum(y1_idx, y2_idx)

        mid_p = 0.5 * (s_pred + e_pred);
        len_p = (e_pred - s_pred + 1.0).clamp(min=1.0)
        mid_t = 0.5 * (s_true + e_true);
        len_t = (e_true - s_true + 1.0).clamp(min=1.0)

        # IoU part
        inter_left = torch.maximum(s_pred, s_true)
        inter_right = torch.minimum(e_pred, e_true)
        inter = torch.clamp(inter_right - inter_left + 1.0, min=0.0)
        union_left = torch.minimum(s_pred, s_true)
        union_right = torch.maximum(e_pred, e_true)
        union = (union_right - union_left + 1.0).clamp(min=1.0)
        iou = inter / union

        # affinity on centers and lengths (index units);  c is a soft scale (≈ 3 index bins)
        c = 3.0
        aff = torch.exp(- (torch.abs(mid_p - mid_t) / c + torch.abs(len_p - len_t) / c))
        aeiou = iou * aff
        return 1.0 - aeiou  # loss

    def _soft_ce_gaussian(self, logits, target_idx, sigma_idx=1.25):
        # logits: [B,T], target_idx: [B] long; returns mean soft CE
        B, T = logits.shape
        with torch.no_grad():
            grid = torch.arange(T, device=logits.device).float().unsqueeze(0)  # [1,T]
            centers = target_idx.float().unsqueeze(1)  # [B,1]
            # normalized Gaussian over indices
            tgt = torch.exp(-0.5 * ((grid - centers) / sigma_idx) ** 2)
            tgt = tgt / (tgt.sum(dim=1, keepdim=True) + 1e-9)  # [B,T]
        logp = F.log_softmax(logits, dim=-1)
        loss = -(tgt * logp).sum(dim=1).mean()
        return loss

    def _soft_ce_gaussian_masked(self, logits, target_idx, sigma_idx=1.25, valid_mask=None):
        B, T = logits.shape
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)

        masked_logits = logits.masked_fill(~valid_mask, -1e9)

        with torch.no_grad():
            grid = torch.arange(T, device=logits.device).float().unsqueeze(0)  # [1,T]
            centers = target_idx.float().unsqueeze(1)  # [B,1]

            # allow vector or scalar sigma
            if torch.is_tensor(sigma_idx):
                sig = sigma_idx.view(-1, 1).clamp_min(1e-6)  # [B,1]
            else:
                sig = torch.tensor(float(sigma_idx), device=logits.device).view(1, 1)

            tgt = torch.exp(-0.5 * ((grid - centers) / sig) ** 2)  # [B,T]
            tgt = tgt * valid_mask.float()
            Z = tgt.sum(dim=1, keepdim=True).clamp_min(1e-9)
            tgt = tgt / Z

        logp = F.log_softmax(masked_logits, dim=-1)
        return -(tgt * logp).sum(dim=1).mean()

    def _interval_prob_from_logits(self, logits_s: torch.Tensor, logits_e: torch.Tensor, temp: float = 0.7):
        """
        Build a soft probability that each year-index y is inside the predicted interval.
        p_in[y] := P(S <= y <= E) ≈ CDF_start_le[y] * CDF_end_ge[y]
        where CDF_start_le[y] = P(S <= y), CDF_end_ge[y] = P(E >= y).

        Returns: [B, T] tensor with values in [0,1]
        """
        # sharpen a bit for crisper intervals
        ps = F.softmax(logits_s / temp, dim=-1)  # [B, T]
        pe = F.softmax(logits_e / temp, dim=-1)  # [B, T]

        # CDFs
        cdf_le = torch.cumsum(ps, dim=-1)  # [B, T], P(S <= y)
        cdf_ge = torch.cumsum(pe.flip(dims=[-1]), dim=-1).flip(dims=[-1])  # [B, T], P(E >= y)

        return (cdf_le * cdf_ge).clamp(0.0, 1.0)  # [B, T]

    def on_train_epoch_start(self):
        # anneal gate temperature from ~1.6 → 0.8 over training
        T = max(1, int(self.max_epochs))
        e = int(self.current_epoch)
        start_T = float(getattr(self, "gate_temp_start", 1.6))
        end_T = float(getattr(self, "gate_temp_end", 0.8))
        self.gate_temp = end_T + (start_T - end_T) * max(0.0, 1.0 - e / T)
        return

    def forward(self, h_idx, r_idx, t_idx):
        h = self.ent_emb(h_idx)         # Dihedron(h)
        r = self.rel_emb(r_idx)         # Dihedron(r)
        t = self.ent_emb(t_idx)         # Dihedron(t)

        # tiny Gaussian noise on embeddings during training (regularization)
        if self.training and self.emb_noise > 0.0:
            n = float(self.emb_noise)
            h = h + n * torch.randn_like(h)
            r = r + n * torch.randn_like(r)
            t = t + n * torch.randn_like(t)

        feats = [h, r, t]
        if self.use_interaction:
            feats.append(torch.abs(h - t))  # |h - t|
        if self.use_prod:
            feats.append(h * t)  # h ⊙ t

        x = torch.cat(feats, dim=1)  # [B, in_feats]
        z = self.trunk(x)  # [B, H]
        B = z.size(0)
        E = self.num_experts

        # --- MoE gate (instance-aware, top-k) ---
        gate_in = self.rel_gate_emb(r_idx) + self.gate_feat(z)  # [B,32]
        gate_logits = self.gate(gate_in) / max(1e-6, float(self.gate_temp))  # [B,E]
        k = int(self.k_experts)
        if k >= E:
            w = F.softmax(gate_logits, dim=-1)  # [B,E]
        else:
            topk_vals, topk_idx = torch.topk(gate_logits, k=k, dim=-1)  # [B,k]
            w = torch.full_like(gate_logits, -1e9)
            w.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            w = F.softmax(w, dim=-1)  # [B,E]

        # --- MoE regression path (mix experts’ outputs) ---
        starts, ends = [], []
        for e in range(E):
            o_e = self.exp_out2[e](z)  # [B,2]
            s_raw_e, d_raw_e = o_e.unbind(dim=-1)
            s_e = torch.sigmoid(s_raw_e)
            d_e = torch.sigmoid(d_raw_e)
            e_e = s_e + (1.0 - s_e) * d_e  # ensure end>=start
            starts.append(s_e.unsqueeze(-1))  # [B,1]
            ends.append(e_e.unsqueeze(-1))  # [B,1]
        starts = torch.cat(starts, dim=-1)  # [B,E]
        ends = torch.cat(ends, dim=-1)  # [B,E]
        start_norm = (starts * w).sum(dim=-1)  # [B]
        end_norm = (ends * w).sum(dim=-1)  # [B]
        preds_norm = torch.stack([start_norm, end_norm], dim=-1)  # [B,2]

        # to index space (float), apply per-relation bias, optional band clamp
        start_idx_f = self._to_index(start_norm)
        end_idx_f = self._to_index(end_norm)

        rb_s = self.rel_bias_start[r_idx.long()]
        rb_e = self.rel_bias_end[r_idx.long()]
        start_idx_f = start_idx_f + rb_s
        end_idx_f = end_idx_f + rb_e

        if self.use_bands and self._ranges_frozen:
            rmin = self.rel_min_idx[r_idx.long()]
            rmax = self.rel_max_idx[r_idx.long()]
            start_idx_f = torch.max(start_idx_f, rmin)
            start_idx_f = torch.min(start_idx_f, rmax)
            end_idx_f = torch.max(end_idx_f, rmin)
            end_idx_f = torch.min(end_idx_f, rmax)

        # back to [0,1]
        start_norm = self._to_norm(start_idx_f)
        end_norm = self._to_norm(end_idx_f)
        preds_norm = torch.stack([start_norm, end_norm], dim=-1)

        # --- MoE time heads (weighted mix of expert projections) ---
        q_s_list, q_e_list = [], []
        for e in range(E):
            q_s_list.append(self.exp_proj_s[e](z).unsqueeze(-1))  # [B,d_t,1]
            q_e_list.append(self.exp_proj_e[e](z).unsqueeze(-1))  # [B,d_t,1]
        q_s_stack = torch.cat(q_s_list, dim=-1)  # [B,d_t,E]
        q_e_stack = torch.cat(q_e_list, dim=-1)  # [B,d_t,E]
        q_s = (q_s_stack * w.unsqueeze(1)).sum(dim=-1)  # [B,d_t]
        q_e = (q_e_stack * w.unsqueeze(1)).sum(dim=-1)  # [B,d_t]

        Ecal = self.time_emb.weight  # [T, d_t]
        logits_start = q_s @ Ecal.t()  # [B, T]
        logits_end = q_e @ Ecal.t()  # [B, T]

        if not hasattr(self, "_printed_fw_once"):
            print("[FW] preds_norm shape:", preds_norm.shape,
                  " logits shapes:", logits_start.shape, logits_end.shape)
            self._printed_fw_once = True

        return preds_norm, (logits_start, logits_end), (q_s, q_e), w

    def _loss_on_norm(self, preds_norm, y1_idx, y2_idx):
        """
        BASELINE: compute loss in INDEX space (not year-normalized).
        """
        start_norm, end_norm = preds_norm[:, 0], preds_norm[:, 1]

        # normalized -> float index
        start_idx_f = self._to_index(start_norm)
        end_idx_f = self._to_index(end_norm)

        # main loss directly on indices
        loss = self.loss_fn(start_idx_f, y1_idx.float()) + self.end_weight * self.loss_fn(end_idx_f, y2_idx.float())

        # return shapes compatible with training_step's logging (we won't use y*_norm now)
        # fabricate "norms" for logging consistency (not used for loss terms now)
        y1_norm = self._to_norm(y1_idx.float())
        y2_norm = self._to_norm(y2_idx.float())

        return loss, start_norm, end_norm, y1_norm, y2_norm

    def training_step(self, batch, batch_idx):
        # unpack batch
        h, r, t, y1_idx, y2_idx = batch
        y1 = y1_idx.long()
        y2 = y2_idx.long()

        # ensure y_start_true <= y_end_true for all rows (guards against swapped labels)
        y_start_true = torch.minimum(y1, y2)
        y_end_true = torch.maximum(y1, y2)

        # ---- update observed per-relation bands (TRAIN only) ----
        with torch.no_grad():
            uniq_r = torch.unique(r)
            for rid in uniq_r:
                m = (r == rid)
                if m.any():
                    rid_i = int(rid.item())
                    min_y = y_start_true[m].float().min()
                    max_y = y_end_true[m].float().max()
                    self.rel_min_obs[rid_i] = torch.minimum(self.rel_min_obs[rid_i], min_y)
                    self.rel_max_obs[rid_i] = torch.maximum(self.rel_max_obs[rid_i], max_y)

        # ---- relation-aware duration prior: EMA update + loss ----
        dur_true = (y_end_true - y_start_true).float()  # in index units
        if self.use_prior:
            with torch.no_grad():
                for rid in uniq_r:
                    m = (r == rid)
                    if m.any():
                        rid_i = int(rid.item())
                        batch_mu = dur_true[m].mean()
                        batch_sd = dur_true[m].std(unbiased=False).clamp_min(1.0)  # avoid 0
                        alpha = float(self.prior_momentum)
                        self.rel_mu[rid_i] = (1 - alpha) * self.rel_mu[rid_i] + alpha * batch_mu
                        self.rel_sigma[rid_i] = (1 - alpha) * self.rel_sigma[rid_i] + alpha * batch_sd
                        self.rel_count[rid_i] = self.rel_count[rid_i] + m.sum()
        else:
            pass
        # predicted duration (index space) from the regression head we already built
        # (we'll reuse start_idx_f / end_idx_f after preds_norm is computed)

        # forward: dual-head
        preds_norm, (logits_s, logits_e), (q_s, q_e), w = self.forward(h, r, t)  # w: [B,E]


        start_norm, end_norm = preds_norm[:, 0], preds_norm[:, 1]

        # Head-consistency: soft-argmax of logits should match regression head indices
        idxs = torch.arange(self.num_times, device=logits_s.device, dtype=torch.float)
        ps_soft = (F.softmax(logits_s, dim=-1) * idxs).sum(dim=-1)
        pe_soft = (F.softmax(logits_e, dim=-1) * idxs).sum(dim=-1)

        start_idx_f = self._to_index(start_norm)
        end_idx_f = self._to_index(end_norm)

        if self.use_prior:
            dur_pred = (end_idx_f - start_idx_f)
            mu_r = self.rel_mu[r].detach()
            sd_r = self.rel_sigma[r].detach().clamp_min(1.0)
            z = (dur_pred - mu_r) / sd_r
            loss_prior = (z * z).mean()
        else:
            loss_prior = torch.tensor(0.0, device=self.device)

        consistency = F.l1_loss(ps_soft, start_idx_f.detach()) + F.l1_loss(pe_soft, end_idx_f.detach())

        # ---- (A) regression-on-index loss (scalar) ----
        # if your self._loss_on_norm already returns a scalar loss, keep it
        loss_reg, _, _, _, _ = self._loss_on_norm(preds_norm, y1_idx, y2_idx)
        # ensure it's scalar
        if loss_reg.dim() != 0:
            loss_reg = loss_reg.mean()

        sigma_idx = getattr(self, "gauss_sigma_idx", 1.25)

        if self.use_bands:
            T = logits_s.size(1)
            grid = torch.arange(T, device=logits_s.device).unsqueeze(0)  # [1,T]
            rmin = self.rel_min_idx[r.long()].unsqueeze(1)  # [B,1]
            rmax = self.rel_max_idx[r.long()].unsqueeze(1)  # [B,1]

            valid_start = (grid >= rmin) & (grid <= rmax)  # [B,T]

            # row-wise sigma from EMA duration (index units) -> reasonable width for CE
            with torch.no_grad():
                # rel_sigma is in index-space; downscale a bit for CE sharpness
                sigma_row = (self.rel_sigma[r].detach().clamp(1.0, 12.0) * 0.6)  # [B]

            ce_s = self._soft_ce_gaussian_masked(
                logits_s, y_start_true,
                sigma_idx=sigma_row, valid_mask=valid_start
            )

            valid_end = (grid >= torch.maximum(rmin, y_start_true.unsqueeze(1))) & (grid <= rmax)

            # coupling inside the band
            with torch.no_grad():
                logits_s_masked = logits_s.masked_fill(~valid_start, -1e9)
                ps_s_masked = F.softmax(logits_s_masked, dim=-1)
                cdf_le = torch.cumsum(ps_s_masked, dim=-1)
            logits_e_coupled = logits_e + torch.log(cdf_le.clamp_min(1e-6))

            ce_e = self._soft_ce_gaussian_masked(logits_e_coupled, y_end_true,
                                                 sigma_idx=sigma_row, valid_mask=valid_end)
        else:
            # unbanded CE baseline
            ce_s = self._soft_ce_gaussian_masked(logits_s, y_start_true, sigma_idx=sigma_idx, valid_mask=None)
            with torch.no_grad():
                cdf_le = torch.cumsum(F.softmax(logits_s, dim=-1), dim=-1)
            logits_e_coupled = logits_e + torch.log(cdf_le.clamp_min(1e-6))
            T = logits_e.size(1)
            valid_end = torch.arange(T, device=logits_e.device).unsqueeze(0) >= y_start_true.unsqueeze(1)
            ce_e = self._soft_ce_gaussian_masked(logits_e_coupled, y_end_true, sigma_idx=sigma_idx,
                                                 valid_mask=valid_end)

        loss_ce = ce_s + self.end_weight * ce_e

        # ---- (C) tiny end-boost / ordering penalties (scalars) ----
        # 1) end-boost based on argmax ordering
        s_id = torch.argmax(logits_s, dim=-1)  # [B]
        e_id = torch.argmax(logits_e, dim=-1)  # [B]
        loss_end_boost = (e_id < s_id).float().mean()  # scalar

        # 2) soft order penalty on norms (ensures end>=start)
        loss_order = (end_norm < start_norm).float().mean()  # scalar

        # ---- (D) differentiable soft-IoU (no loops) ----
        # Predicted interval probability from logits
        p_in = self._interval_prob_from_logits(logits_s, logits_e, temp=0.5)  # was 0.7; 0.5–0.8 works

        # Build target interval mask [B,T] with 1's between min(y1,y2) and max(y1,y2)
        T = p_in.size(1)
        grid = torch.arange(T, device=p_in.device).unsqueeze(0)  # [1,T]
        left = torch.minimum(y1, y2).unsqueeze(1)
        right = torch.maximum(y1, y2).unsqueeze(1)
        tgt = ((grid >= left) & (grid <= right)).float()

        # IoU = sum min / sum max
        inter = torch.sum(torch.min(p_in, tgt), dim=1)
        union = torch.sum(torch.max(p_in, tgt), dim=1).clamp_min(1e-6)
        soft_iou = inter / union
        loss_iou = (1.0 - soft_iou).mean()

        # If you added gIoU / aeIoU helpers, do the same:
        giou_loss = torch.tensor(0.0, device=self.device)  # replace with your mean(...) if implemented
        aeiou_loss = torch.tensor(0.0, device=self.device)  # replace with your mean(...) if implemented

        align_q = 0.01 * (q_s - q_e).pow(2).mean()

        # MoE monitoring
        gate_entropy = -(w.clamp_min(1e-8) * torch.log(w.clamp_min(1e-8))).sum(dim=-1).mean()
        self.log("moe/entropy", gate_entropy, on_step=False, on_epoch=True, prog_bar=False)

        # Load-balance (encourage using all experts)
        m = w.mean(dim=0)  # [E]
        u = torch.full_like(m, 1.0 / m.numel())
        gate_balance_loss = -(u * torch.log(m.clamp_min(1e-8))).sum()

        bias_reg = self.rel_bias_l2 * (self.rel_bias_start.pow(2).mean() + self.rel_bias_end.pow(2).mean())

        # ---- (E) total loss (must be SCALAR) ----
        #loss = (
         #       loss_reg
          #      + 0.08 * loss_ce
           #     + 0.02 * loss_end_boost
            #    + 0.01 * loss_order
             #   + 0.20 * loss_iou
              #  + 0.00 * giou_loss
         #       + 0.00 * aeiou_loss
          #      + 0.01 * consistency
           #     + 0.01 * align_q
            #    + float(self.prior_weight) * loss_prior
             #   + float(self.gate_balance) * gate_balance_loss
              #  + bias_reg
        #)
        loss = (
                self.w_reg * loss_reg
                + self.w_ce * loss_ce
                + self.w_end_boost * loss_end_boost
                + self.w_order * loss_order
                + self.w_iou * loss_iou
                + 0.00 * giou_loss
                + 0.00 * aeiou_loss
                + self.w_consistency * consistency
                + self.w_align * align_q
                + float(self.prior_weight) * loss_prior
                + float(self.gate_balance) * gate_balance_loss
                + bias_reg
        )

        # === Tiny regularizers on the shared calendar (stability/generalization) ===
        E = self.time_emb.weight
        smooth = (E[1:] - E[:-1]).pow(2).mean()  # Laplacian smoothness across neighboring years
        ridge = E.pow(2).mean()  # L2 on the table
       # loss = loss + 0.010 * smooth + 0.0003 * ridge
        loss = loss + self.w_smooth * smooth + self.w_ridge * ridge

        # safety check: must be scalar
        assert loss.dim() == 0, f"training_step loss must be scalar, got shape {tuple(loss.shape)}"

        # ---- logging ----
        self.log("train_loss", loss, prog_bar=True)

        # (optional) detailed loss breakdown at epoch end
        self.log_dict({
            "l/reg": loss_reg.detach(),
            "l/ce": loss_ce.detach(),
            "l/endB": loss_end_boost.detach(),
            "l/order": loss_order.detach(),
            "l/softIoU": loss_iou.detach(),
            "l/giou": giou_loss.detach(),
            "l/aeiou": aeiou_loss.detach(),
            "l/smooth": smooth.detach(),
            "l/ridge": ridge.detach(),
            "l/align_q": align_q.detach(),
            "l/prior": loss_prior.detach(),
            "moe/balance": gate_balance_loss.detach(),
            "moe/entropy": gate_entropy.detach(),
        }, prog_bar=False, on_step=False, on_epoch=True)

        # ---- human-friendly metrics in YEARS (unchanged) ----
        with torch.no_grad():
            pred_start_years = self._norm_to_year(start_norm).clamp(self.min_year, self.max_year)
            pred_end_years = self._norm_to_year(end_norm).clamp(self.min_year, self.max_year)

            idx2year = self.idx_to_year
            tgt_start_years = torch.tensor([float(idx2year[int(i)]) for i in y1_idx.tolist()], device=self.device)
            tgt_end_years = torch.tensor([float(idx2year[int(i)]) for i in y2_idx.tolist()], device=self.device)

            swap = pred_start_years > pred_end_years
            if swap.any():
                tmp = pred_start_years[swap].clone()
                pred_start_years[swap] = pred_end_years[swap]
                pred_end_years[swap] = tmp

            mae_start = torch.mean(torch.abs(pred_start_years - tgt_start_years))
            mae_end = torch.mean(torch.abs(pred_end_years - tgt_end_years))
            self.log("train_mae_year_start", mae_start, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_mae_year_end", mae_end, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def _calendar_val_loss(self, logits_s, logits_e, y1_idx, y2_idx):
        y_start_true = torch.minimum(y1_idx.long(), y2_idx.long())
        y_end_true = torch.maximum(y1_idx.long(), y2_idx.long())

        sigma_idx = getattr(self, "gauss_sigma_idx", 1.25)

        ce_s = self._soft_ce_gaussian_masked(
            logits_s,
            y_start_true,
            sigma_idx=sigma_idx,
            valid_mask=None
        )

        with torch.no_grad():
            cdf_le = torch.cumsum(F.softmax(logits_s, dim=-1), dim=-1)

        logits_e_coupled = logits_e + torch.log(cdf_le.clamp_min(1e-6))

        T = logits_e.size(1)
        valid_end = torch.arange(T, device=logits_e.device).unsqueeze(0) >= y_start_true.unsqueeze(1)

        ce_e = self._soft_ce_gaussian_masked(
            logits_e_coupled,
            y_end_true,
            sigma_idx=sigma_idx,
            valid_mask=valid_end
        )

        return ce_s + self.end_weight * ce_e

    def validation_step(self, batch, batch_idx):
        h, r, t, y1_idx, y2_idx = batch
        preds_norm, (logits_s, logits_e), _, _ = self.forward(h, r, t)
        if self.pred_head == "calendar" and self.w_reg == 0.0:
            loss = self._calendar_val_loss(logits_s, logits_e, y1_idx, y2_idx)
        else:
            loss, _, _, _, _ = self._loss_on_norm(preds_norm, y1_idx, y2_idx)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # soft-IoU for validation (mirrors train block D)
        with torch.no_grad():
            p_in = self._interval_prob_from_logits(logits_s, logits_e, temp=0.6)
            T = p_in.size(1)
            grid = torch.arange(T, device=p_in.device).unsqueeze(0)
            left = torch.minimum(y1_idx.long(), y2_idx.long()).unsqueeze(1)
            right = torch.maximum(y1_idx.long(), y2_idx.long()).unsqueeze(1)
            tgt = ((grid >= left) & (grid <= right)).float()
            inter = torch.sum(torch.min(p_in, tgt), dim=1)
            union = torch.sum(torch.max(p_in, tgt), dim=1).clamp_min(1e-6)
            val_soft_iou = (inter / union).mean()
        self.log("val_softIoU", val_soft_iou, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        # ---- once: freeze relation bands from observed TRAIN stats ----
        if self.use_bands and (not self._ranges_frozen) and (
                int(self.current_epoch) + 1 >= int(self.band_freeze_epoch)):
            with torch.no_grad():
                # finalize min/max with margin and clamp
                margin = float(self.band_margin)
                rel_min = (self.rel_min_obs - margin).clamp(0.0, float(self.num_times - 1))
                rel_max = (self.rel_max_obs + margin).clamp(0.0, float(self.num_times - 1))
                # ensure min <= max
                rel_min = torch.minimum(rel_min, rel_max)
                self.rel_min_idx.copy_(rel_min)
                self.rel_max_idx.copy_(rel_max)
                self._ranges_frozen = True
                print("[BANDS] Frozen per-relation year bands.")

        if not self.use_swa:
            return

        epoch = int(self.current_epoch)
        if epoch < self.swa_start_epoch:
            return

        # lazy init once we know device/dtype
        if not self._swa_inited:
            from torch.optim.swa_utils import AveragedModel
            self._swa_model = AveragedModel(self)
            self._swa_inited = True

        # update running average every epoch (or every k epochs)
        if (epoch - self.swa_start_epoch) % self.swa_update_freq == 0:
            self._swa_model.update_parameters(self)

    def on_fit_end(self):
        super().on_fit_end()
        # If SWA was used, copy averaged weights into the live model for test/export
        if self.use_swa and self._swa_inited and (self._swa_model is not None):
            for p_avg, p in zip(self._swa_model.parameters(), self.parameters()):
                p.data.copy_(p_avg.data)

    def test_step(self, batch, batch_idx):
        h, r, t, y1_idx, y2_idx = batch
        preds_norm, (logits_s, logits_e), _, _ = self.forward(h, r, t)

        if self.pred_head == "calendar" and self.w_reg == 0.0:
            loss = self._calendar_val_loss(logits_s, logits_e, y1_idx, y2_idx)
        else:
            loss, _, _, _, _ = self._loss_on_norm(preds_norm, y1_idx, y2_idx)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        # read values you already stored in __init__
        lr = float(getattr(self, "lr", 1e-3))
        wd = float(getattr(self, "weight_decay", 0.0))
        max_epochs = int(getattr(self, "max_epochs", 80))

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        # Cosine decay from lr down to 10% of it by the final epoch
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.1)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",  # step each epoch
                "frequency": 1,
            },
        }

    def forward_triples(self, h, r, t, y1, y2, type=None):
        preds_norm, (logits_s, logits_e), _, _ = self.forward(h, r, t)

        if getattr(self, "pred_head", "calendar") == "regression":
            # Use regression head for final prediction
            ps = self._to_index(preds_norm[:, 0])
            pe = self._to_index(preds_norm[:, 1])
            pe = torch.maximum(pe, ps)

        else:
            # Use calendar head for final prediction
            idxs = torch.arange(self.num_times, device=logits_s.device, dtype=torch.float)
            ps = (torch.softmax(logits_s, dim=-1) * idxs).sum(dim=-1)
            pe = (torch.softmax(logits_e, dim=-1) * idxs).sum(dim=-1)
            pe = torch.maximum(pe, ps)

        # Safe clamp to [0, T-1]
        ps = ps.clamp(0, self.num_times - 1)
        pe = pe.clamp(0, self.num_times - 1)

        # Per-relation band clamp at test time, only if bands were frozen from train
        if getattr(self, "_ranges_frozen", False):
            rmin = self.rel_min_idx[r.long()]
            rmax = self.rel_max_idx[r.long()]
            ps = torch.maximum(ps, rmin)
            pe = torch.minimum(pe, rmax)
            pe = torch.maximum(pe, ps)

        return ps, pe

# ===== Simple inference helpers for exporting predictions =====
    @torch.no_grad()
    def predict_indices(self, h_idx, r_idx, t_idx, use_soft=False):
        """
        Inputs:
          h_idx, r_idx, t_idx: 1D LongTensor of indices (same id space used in training)
        Returns:
          start_idx_long, end_idx_long  (1D LongTensors), clamped to [0, num_times-1] and with start<=end
        """
        # Ensure tensors and device
        device = next(self.parameters()).device

        def to_long(x):
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            return x.to(device=device, dtype=torch.long).view(-1)

        h_idx = to_long(h_idx)
        r_idx = to_long(r_idx)
        t_idx = to_long(t_idx)

        # Forward to get logits over calendar indices
        _, (logits_s, logits_e), _, _ = self.forward(h_idx, r_idx, t_idx)  # [B,T] each

        T = int(self.num_times)
        if use_soft:
            # soft expectation over indices, then round to nearest index
            grid = torch.arange(T, device=logits_s.device, dtype=torch.float)
            ps = (torch.softmax(logits_s, dim=-1) * grid).sum(dim=-1).round()
            pe = (torch.softmax(logits_e, dim=-1) * grid).sum(dim=-1).round()
        else:
            # hard argmax over logits (discrete index)
            ps = torch.argmax(logits_s, dim=-1).to(torch.long)
            pe = torch.argmax(logits_e, dim=-1).to(torch.long)

        # Clamp to valid range and enforce order
        ps = ps.clamp(0, T - 1)
        pe = pe.clamp(0, T - 1)
        swap = ps > pe
        if swap.any():
            tmp = ps[swap].clone()
            ps[swap] = pe[swap]
            pe[swap] = tmp

        return ps.to(torch.long), pe.to(torch.long)

    @torch.no_grad()
    def predict_years(self, h_idx, r_idx, t_idx, use_soft=False):
        """
        Same as predict_indices but returns YEARS (Python ints) using self.idx_to_year.
        """
        ps, pe = self.predict_indices(h_idx, r_idx, t_idx, use_soft=use_soft)
        # map indices -> years using the table constructed in __init__
        idx2year = self.idx_to_year  # list[int], length = num_times
        years_s = [int(idx2year[int(i)]) for i in ps.cpu().tolist()]
        years_e = [int(idx2year[int(i)]) for i in pe.cpu().tolist()]
        return years_s, years_e