"""Unified semantic and instance head over superpoints.

Ported from the original notebook with the backbone made explicit and the
per-scene padding loop replaced by a scatter. Behaviour is otherwise unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reify.models.backbone import Backbone


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[1]))
    out.index_add_(0, index, src)
    count = src.new_zeros((dim_size, 1))
    count.index_add_(0, index, src.new_ones((src.shape[0], 1)))
    return out / count.clamp_min(1.0)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, dim_ff: int = 512, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_key_padding_mask=None):
        q2, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.n1(q + q2)
        q2, _ = self.cross_attn(q, kv, kv, key_padding_mask=kv_key_padding_mask, need_weights=False)
        q = self.n2(q + q2)
        q = self.n3(q + self.ffn(q))
        return q


class OneFormer3DLike(nn.Module):
    def __init__(
        self,
        num_classes: int = 20,
        in_channels: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        k_ins: int = 256,
        query_aug_std: float = 0.02,
        backbone: str = "auto",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.k_ins = k_ins
        self.query_aug_std = query_aug_std

        self.backbone = Backbone(in_ch=in_channels, out_ch=d_model, kind=backbone)
        self.sem_queries = nn.Embedding(num_classes, d_model)
        self.inst_bias = nn.Embedding(k_ins, d_model)
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)]
        )
        self.inst_cls = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.inst_kernel = nn.Linear(d_model, d_model)
        self.sem_kernel = nn.Linear(d_model, d_model)

    @property
    def backbone_kind(self) -> str:
        return self.backbone.kind

    @torch.no_grad()
    def _sample_queries(self, m_list: list[int], device) -> torch.Tensor:
        b, k = len(m_list), self.k_ins
        src = torch.zeros((b, k), dtype=torch.long, device=device)
        for i, m in enumerate(m_list):
            if m <= 0:
                continue
            if m >= k:
                src[i] = torch.randperm(m, device=device)[:k]
            else:
                src[i] = torch.randint(0, m, (k,), device=device)
        return src

    def forward(self, batch: dict) -> dict:
        device = batch["feats"].device
        b, m_max = batch["sp_mask"].shape
        offsets = batch["sp_offsets"]
        m_list = [int(offsets[i + 1] - offsets[i]) for i in range(b)]

        vox_feat = self.backbone(batch["coords"], batch["feats"])
        sum_m = int(offsets[-1].item())
        sp_all = scatter_mean(vox_feat, batch["vox_sp_global"], dim_size=sum_m)

        # pad (sum_m, D) to (B, m_max, D) with a scatter rather than a Python loop
        d = sp_all.shape[1]
        sp_feats = torch.zeros((b, m_max, d), dtype=sp_all.dtype, device=device)
        rows = torch.arange(sum_m, device=device)
        batch_of = torch.bucketize(rows, offsets[1:], right=True).clamp_max(b - 1)
        local = rows - offsets[batch_of]
        sp_feats[batch_of, local] = sp_all

        sp_mask = batch["sp_mask"]
        pad_mask = ~sp_mask

        src_sp = self._sample_queries(m_list, device)
        inst_q = sp_feats.gather(1, src_sp.unsqueeze(-1).expand(b, self.k_ins, d))

        if self.training and self.query_aug_std > 0:
            half = (torch.rand((b, self.k_ins), device=device) < 0.5).unsqueeze(-1).float()
            inst_q = inst_q + torch.randn_like(inst_q) * self.query_aug_std * half

        inst_q = inst_q + self.inst_bias(torch.arange(self.k_ins, device=device)).unsqueeze(0)
        sem_q = self.sem_queries.weight.unsqueeze(0).expand(b, -1, -1)

        q = torch.cat([inst_q, sem_q], dim=1)
        for layer in self.decoder:
            q = layer(q, sp_feats, kv_key_padding_mask=pad_mask)

        inst_out, sem_out = q[:, : self.k_ins], q[:, self.k_ins :]
        inst_mask_logits = torch.bmm(self.inst_kernel(inst_out), sp_feats.transpose(1, 2))
        sem_mask_logits = torch.bmm(self.sem_kernel(sem_out), sp_feats.transpose(1, 2))

        neg = torch.finfo(inst_mask_logits.dtype).min / 2
        inst_mask_logits = inst_mask_logits.masked_fill(pad_mask.unsqueeze(1), neg)
        sem_mask_logits = sem_mask_logits.masked_fill(pad_mask.unsqueeze(1), neg)

        return {
            "sp_feats": sp_feats,
            "inst_logits": self.inst_cls(inst_out),
            "inst_mask_logits": inst_mask_logits,
            "sem_mask_logits": sem_mask_logits,
            "src_sp": src_sp,
            "m_list": m_list,
        }
