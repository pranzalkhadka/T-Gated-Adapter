"""Temporal CLIPSeg adapter model."""

import torch
import torch.nn as nn
from transformers import CLIPSegForImageSegmentation


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * torch.floor(random_tensor + keep_prob) / keep_prob


class TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=src_key_padding_mask)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class SpatialContextBlock(nn.Module):
    def __init__(self, dim, nhead=8, dropout=0.1, drop_path=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(self.drop(attn_out))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CLIPSegTemporalAdapter(nn.Module):
    def __init__(self, context_size=5, nhead=8, n_layers=4, dropout=0.1, proj_dim=256, drop_path_rate=0.1):
        super().__init__()
        self.context_size = context_size
        self.proj_dim = proj_dim

        self.base_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.base_model.clip.requires_grad_(True)
        self.base_model.decoder.requires_grad_(True)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 352, 352)
            v_out = self.base_model.clip.vision_model(pixel_values=dummy, return_dict=True)
            self.seq_len = v_out.last_hidden_state.shape[1]
            self.Dv = v_out.last_hidden_state.shape[2]

        self.proj_in = nn.Sequential(nn.Linear(self.Dv, proj_dim, bias=False), nn.LayerNorm(proj_dim))
        self.proj_out = nn.Linear(proj_dim, self.Dv, bias=False)
        self.temporal_pos_emb = nn.Parameter(torch.empty(1, context_size, proj_dim))
        nn.init.normal_(self.temporal_pos_emb, std=0.02)

        drop_path_rates = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.temporal_layers = nn.ModuleList(
            [
                TemporalEncoderLayer(
                    d_model=proj_dim,
                    nhead=nhead,
                    dim_feedforward=proj_dim * 4,
                    dropout=dropout,
                    drop_path=drop_path_rates[i],
                )
                for i in range(n_layers)
            ]
        )
        self.transformer_norm = nn.LayerNorm(proj_dim)

        self.gate_proj = nn.Linear(self.Dv, self.Dv, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -5.0)

        self.spatial_context = SpatialContextBlock(self.Dv, nhead=8, dropout=dropout, drop_path=0.05)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.Dv, self.Dv),
            nn.LayerNorm(self.Dv),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.Dv, self.Dv),
            nn.Dropout(dropout * 0.5),
        )

    def _run_temporal_transformer(self, x):
        for layer in self.temporal_layers:
            x = layer(x)
        return self.transformer_norm(x)

    def forward(self, pixel_values, input_ids, attention_mask):
        bsz, ctx, channels, height, width = pixel_values.shape
        pv_flat = pixel_values.view(bsz * ctx, channels, height, width)
        tokens = self.base_model.clip.vision_model(pixel_values=pv_flat, return_dict=True).last_hidden_state
        tokens = tokens.view(bsz, ctx, self.seq_len, self.Dv)

        x = tokens.permute(0, 2, 1, 3).contiguous().view(bsz * self.seq_len, ctx, self.Dv)
        x_proj = self.proj_in(x) + self.temporal_pos_emb
        x_trans = self._run_temporal_transformer(x_proj)
        x_trans = self.proj_out(x_trans).view(bsz, self.seq_len, ctx, self.Dv)

        temporal_feat = x_trans[:, :, ctx // 2, :]
        center_raw = tokens[:, ctx // 2, :, :]
        gate = torch.sigmoid(self.gate_proj(temporal_feat))
        center_tokens = gate * temporal_feat + (1.0 - gate) * center_raw
        center_tokens = self.spatial_context(center_tokens)
        center_tokens = self.mlp_head(center_tokens)

        text_out = self.base_model.clip.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_pooled = text_out.pooler_output

        proj = getattr(self.base_model.clip, "text_projection", None)
        if proj is None:
            text_emb = text_pooled
        elif isinstance(proj, nn.Linear):
            text_emb = proj(text_pooled)
        elif isinstance(proj, (torch.Tensor, nn.Parameter)):
            try:
                text_emb = text_pooled @ proj
            except Exception:
                text_emb = text_pooled.matmul(proj.T)
        elif callable(proj):
            text_emb = proj(text_pooled)
        else:
            text_emb = text_pooled

        decoder_out = self.base_model.decoder(hidden_states=[center_tokens], conditional_embeddings=text_emb)
        logits = decoder_out.logits
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        return logits


def gate_sparsity_loss(model, lambda_gate=0.001):
    raw_model = model.module if hasattr(model, "module") else model
    gate_vals = torch.sigmoid(raw_model.gate_proj.bias)
    entropy = gate_vals * (1.0 - gate_vals)
    return lambda_gate * entropy.mean()


def get_raw_model(model):
    return model.module if hasattr(model, "module") else model
