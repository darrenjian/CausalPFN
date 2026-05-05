"""
temporal_encoder.py — Temporal Encoder E_phi for Irregular Clinical Time-Series

Maps a patient's irregular observation history H_i = {(s_il, x_il)} to a
fixed-size embedding z_i in R^99. The 99-dimensional output matches the slot
available in CausalPFN's input (which prepends treatment t, making it 100 total).

Architecture:
    1. Continuous-time positional encoding: PE(s)_j = sin(s * C^{-j/d_pe})
    2. Input projection: [x_il ; PE(s_il)] -> R^d_model
    3. Transformer encoder layers with multi-head self-attention
    4. Mean pooling over the sequence dimension -> R^d_model
    5. Output projection: R^d_model -> R^99
    6. BatchNorm1d(99) output normalization — maps z_i to approximately
       zero mean, unit variance before passing to CausalPFN's frozen layers.

BatchNorm behaves differently in train vs eval mode. Always call
encoder.eval() before evaluation and encoder.train() before training.

Usage:
    from temporal_encoder import TemporalEncoder, collate_histories
    encoder = TemporalEncoder(n_covariates=5, d_model=64)
    z = encoder(timestamps, covariates, mask)  # z: (batch, 99), normalized
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContinuousTimePositionalEncoding(nn.Module):
    """
    Continuous-time positional encoding:
        PE(s)_j = sin(s * C^{-j/d_pe})    for j = 0, ..., d_pe-1

    Identical to temporal_encoder.py — no changes.
    """

    def __init__(self, d_pe: int, C: float = 100.0):
        super().__init__()
        self.d_pe = d_pe
        self.C = C
        j = torch.arange(d_pe, dtype=torch.float32)
        freqs = C ** (-j / d_pe)
        self.register_buffer("freqs", freqs.unsqueeze(0).unsqueeze(0))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        t = timestamps.unsqueeze(-1)
        return torch.sin(t * self.freqs)


class TemporalEncoder(nn.Module):
    """
    Temporal encoder: irregular clinical histories -> fixed 99-dim embeddings.

    Parameters
    ----------
    n_covariates : int
        Number of time-varying features per observation.
    d_pe : int
        Positional encoding dimensionality.
    d_model : int
        Internal transformer dimension.
    n_heads : int
        Number of attention heads. Must divide d_model.
    n_layers : int
        Transformer depth.
    dropout : float
        Dropout rate.
    """

    D_OUT = 99

    def __init__(
        self,
        n_covariates: int,
        d_pe: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.n_covariates = n_covariates
        self.d_pe = d_pe
        self.d_model = d_model

        # Continuous-time positional encoding (unchanged)
        self.pe = ContinuousTimePositionalEncoding(d_pe=d_pe)

        # Input projection (unchanged)
        self.input_proj = nn.Linear(n_covariates + d_pe, d_model)

        # Transformer encoder (unchanged)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        self.output_proj = nn.Linear(d_model, self.D_OUT)
        self.output_norm = nn.BatchNorm1d(self.D_OUT)
        self.reconstruction_head = nn.Linear(d_model, n_covariates)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.reconstruction_head.weight, std=0.02)
        nn.init.zeros_(self.reconstruction_head.bias)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)

    def _pool(self, token_embeddings, mask):
        """Mean pool over non-padding positions."""
        real_mask_f = (~mask).float().unsqueeze(-1)
        pooled = (token_embeddings * real_mask_f).sum(dim=1)
        pooled = pooled / real_mask_f.sum(dim=1).clamp(min=1.0)
        return pooled

    def encode(
        self,
        timestamps: torch.Tensor,
        covariates: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode irregular histories -> per-token representations."""
        pe = self.pe(timestamps)
        x = torch.cat([covariates, pe], dim=-1)
        x = self.input_proj(x)
        return self.transformer(x, src_key_padding_mask=mask)

    def forward(
        self,
        timestamps: torch.Tensor,
        covariates: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Irregular history -> normalized 99-dim embedding z_i."""
        z, _ = self.forward_with_token_embeddings(timestamps, covariates, mask)
        return z

    def forward_with_token_embeddings(
        self,
        timestamps: torch.Tensor,
        covariates: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, token_embeddings) for reconstruction loss."""
        token_embeddings = self.encode(timestamps, covariates, mask)
        pooled = self._pool(token_embeddings, mask)
        z = self.output_proj(pooled)
        z = self.output_norm(z)
        return z, token_embeddings

    def truncated_forward(
        self,
        timestamps: torch.Tensor,
        covariates: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode history with last real observation removed."""
        real_mask = ~mask
        seq_len = timestamps.shape[1]
        positions = torch.arange(seq_len, device=timestamps.device).unsqueeze(0)
        last_real_idx = (real_mask.long() * positions).argmax(dim=1)
        truncated_mask = mask.clone()
        batch_idx = torch.arange(mask.shape[0], device=mask.device)
        truncated_mask[batch_idx, last_real_idx] = True
        return self.forward(timestamps, covariates, truncated_mask)


# ------------------------------------------------------------------
# Collation
# ------------------------------------------------------------------

def collate_histories(
    histories,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a list of PatientHistory objects into padded tensors."""
    batch_size = len(histories)
    max_seq = max(h.n_obs for h in histories)
    n_cov = histories[0].covariates.shape[1]

    timestamps_tensor = torch.zeros(batch_size, max_seq, device=device)
    covariates_tensor = torch.zeros(batch_size, max_seq, n_cov, device=device)
    mask = torch.ones(batch_size, max_seq, dtype=torch.bool, device=device)

    for i, h in enumerate(histories):
        n = h.n_obs
        timestamps_tensor[i, :n] = torch.from_numpy(h.timestamps).to(device)
        covariates_tensor[i, :n] = torch.from_numpy(h.covariates).to(device)
        mask[i, :n] = False

    return timestamps_tensor, covariates_tensor, mask


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dgp import make_dgp

    print("Running TemporalEncoder sanity check (with BatchNorm)...")

    device = torch.device("cpu")
    dgp   = make_dgp("strong_temporal", n_covariates=5, seed=42)
    batch = dgp.sample_batch(n_patients=16, window_hours=48.0)

    encoder = TemporalEncoder(
        n_covariates=5, d_pe=16, d_model=64, n_heads=4, n_layers=2,
    ).to(device)

    timestamps, covariates, mask = collate_histories(batch.histories, device)

    # Train mode: BatchNorm uses batch statistics
    encoder.train()
    z_train = encoder(timestamps, covariates, mask)
    print(f"  z shape (train):  {z_train.shape}  (should be (16, 99))")
    print(f"  z mean  (train):  {z_train.mean().item():.4f}  (near 0 after norm)")
    print(f"  z std   (train):  {z_train.std().item():.4f}   (near 1 after norm)")
    assert z_train.shape == (16, 99)

    # Eval mode: BatchNorm uses running statistics
    encoder.eval()
    with torch.no_grad():
        z_eval = encoder(timestamps, covariates, mask)
    print(f"  z shape (eval):   {z_eval.shape}")

    # Token embeddings
    encoder.train()
    z2, token_emb = encoder.forward_with_token_embeddings(timestamps, covariates, mask)
    assert z2.shape == z_train.shape
    print(f"  Token embeddings: {token_emb.shape}")

    # Truncated forward
    z_trunc = encoder.truncated_forward(timestamps, covariates, mask)
    assert z_trunc.shape == (16, 99)
    print(f"  Truncated z:      {z_trunc.shape}")

    assert not torch.isnan(z_train).any(), "NaN in embeddings"
    assert not torch.isinf(z_train).any(), "Inf in embeddings"

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    print("\nTemporalEncoder sanity check passed.")
