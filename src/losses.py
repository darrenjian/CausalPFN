"""
losses.py — Self-Supervised Loss Terms for the Temporal Encoder

The causal loss (L_causal) comes directly from icl_model.forward() and
is computed in the training loop. This file provides the two remaining
components of the joint objective:

    L_total = lambda1 * L_causal
            + lambda2 * L_reconstruction
            + lambda3 * L_consistency

L_reconstruction (Self-Supervised — Masked Feature Prediction)
    Randomly masks a fraction of observations in each patient's history.
    The encoder must reconstruct the masked covariate values from context.
    Inspired by BERT's masked language modeling, applied to clinical time-series.

    Why this matters: forces the encoder to understand the geometry of the
    data distribution, not just the features that predict outcomes in our
    synthetic DGP. This is the primary defense against the DGP-coverage gap
    — the encoder remains grounded in real clinical data patterns.

L_consistency (Structural Regularizer — Temporal Smoothness)
    Penalizes discontinuous jumps in embedding space when a new observation
    is added. The embedding of the full history should be a smooth update
    of the embedding without the last observation.

    Uses stop_gradient on the truncated embedding to avoid the trivial
    solution of collapsing all embeddings to zero.

    Clinical motivation: a patient's physiological state evolves continuously.
    Their embedding should reflect this continuity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def masked_reconstruction_loss(
    encoder: nn.Module,
    timestamps: torch.Tensor,
    covariates: torch.Tensor,
    mask: torch.Tensor,
    mask_fraction: float = 0.15,
) -> torch.Tensor:
    """
    Compute the masked feature reconstruction loss.

    Randomly masks mask_fraction of real observations, runs the encoder,
    uses the token-level representations at masked positions to predict
    the original covariate values, and computes MSE loss.

    Parameters
    ----------
    encoder : TemporalEncoder
        The temporal encoder with a reconstruction_head attribute.
    timestamps : torch.Tensor of shape (batch, max_seq)
    covariates : torch.Tensor of shape (batch, max_seq, n_covariates)
    mask : torch.Tensor of shape (batch, max_seq), dtype=bool
        True = padding position. This is the existing padding mask.
    mask_fraction : float
        Fraction of real observations to mask for reconstruction.
        Default 0.15 matches BERT's original masking rate.

    Returns
    -------
    loss : torch.Tensor scalar
        MSE between predicted and actual covariate values at masked positions.
    """
    batch_size, max_seq, n_cov = covariates.shape
    device = covariates.device

    # Real positions: True where NOT padding
    real_positions = ~mask  # (batch, max_seq)

    # Sample reconstruction mask: mask_fraction of real positions
    # We sample independently per position using a uniform draw
    rand = torch.rand(batch_size, max_seq, device=device)

    # A position is masked for reconstruction if:
    # (1) it is a real observation AND (2) random draw < mask_fraction
    recon_mask = real_positions & (rand < mask_fraction)  # (batch, max_seq)

    # If no positions are masked in a sample (can happen with very short histories),
    # force-mask the second position as a fallback
    no_masked = ~recon_mask.any(dim=1)  # (batch,)
    if no_masked.any():
        # Find the second real position (index 1) for those samples
        real_cumsum = real_positions.long().cumsum(dim=1)
        second_pos = (real_cumsum == 2).float().argmax(dim=1)  # (batch,)
        for b in no_masked.nonzero(as_tuple=True)[0]:
            recon_mask[b, second_pos[b]] = True

    # Save original covariate values at masked positions before zeroing
    # target shape: (n_masked_total, n_cov)
    masked_idx = recon_mask.nonzero(as_tuple=False)  # (n_masked, 2)
    target_covariates = covariates[masked_idx[:, 0], masked_idx[:, 1]]  # (n_masked, n_cov)

    # Zero out masked positions in the input (masking strategy: replace with 0)
    # Also zero out their timestamps to remove temporal signal at masked positions
    masked_covariates = covariates.clone()
    masked_timestamps = timestamps.clone()
    masked_covariates[recon_mask] = 0.0
    masked_timestamps[recon_mask] = 0.0

    # Build the key_padding_mask for the transformer:
    # mask out both original padding AND reconstruction-masked positions
    combined_mask = mask | recon_mask  # (batch, max_seq)

    # Forward pass through encoder (getting token-level embeddings)
    _, token_embeddings = encoder.forward_with_token_embeddings(
        masked_timestamps, masked_covariates, combined_mask
    )
    # token_embeddings: (batch, max_seq, d_model)

    # Extract embeddings at masked positions
    masked_token_emb = token_embeddings[masked_idx[:, 0], masked_idx[:, 1]]
    # (n_masked, d_model)

    # Predict original covariate values using reconstruction head
    predicted_covariates = encoder.reconstruction_head(masked_token_emb)
    # (n_masked, n_cov)

    # MSE loss between predicted and actual values
    loss = F.mse_loss(predicted_covariates, target_covariates)
    return loss


def temporal_consistency_loss(
    encoder: nn.Module,
    timestamps: torch.Tensor,
    covariates: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the temporal consistency loss.

    Enforces that the encoder's output changes smoothly as new observations
    are added. Specifically:

        L_consistency = || z_full - stop_gradient(z_truncated) ||^2

    where z_full is the embedding of the complete history and z_truncated
    is the embedding with the last real observation removed.

    The stop_gradient on z_truncated means gradients only flow through
    the full-history branch. This prevents the trivial solution of
    collapsing all embeddings to zero to minimize the L2 distance.

    Parameters
    ----------
    encoder : TemporalEncoder
    timestamps : torch.Tensor of shape (batch, max_seq)
    covariates : torch.Tensor of shape (batch, max_seq, n_covariates)
    mask : torch.Tensor of shape (batch, max_seq), dtype=bool

    Returns
    -------
    loss : torch.Tensor scalar
        Mean squared L2 distance between full and truncated embeddings.
    """
    # Full history embedding — gradients flow through this
    z_full = encoder(timestamps, covariates, mask)  # (batch, 99)

    # Truncated embedding — stop gradient so only z_full is updated
    with torch.no_grad():
        z_truncated = encoder.truncated_forward(timestamps, covariates, mask)
    # stop_gradient: detach so no gradients flow back through z_truncated
    z_truncated = z_truncated.detach()

    # L2 distance between full and truncated embeddings
    # Mean over batch and embedding dimensions
    loss = F.mse_loss(z_full, z_truncated)
    return loss


class JointLoss(nn.Module):
    """
    Combined loss module for the full training objective:

        L_total = lambda1 * L_causal
                + lambda2 * L_reconstruction
                + lambda3 * L_consistency

    Wraps the three loss terms with configurable weights and provides
    logging of individual components for monitoring training dynamics.

    Parameters
    ----------
    lambda1 : float
        Weight on the causal loss (supervised). Default 1.0.
    lambda2 : float
        Weight on the reconstruction loss (self-supervised). Default 0.5.
        Slightly down-weighted relative to causal to keep causal signal dominant.
    lambda3 : float
        Weight on the consistency loss (structural). Default 0.1.
        Light regularizer — should not dominate.
    mask_fraction : float
        Fraction of observations to mask for reconstruction loss.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda3: float = 0.1,
        mask_fraction: float = 0.15,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.mask_fraction = mask_fraction

    def forward(
        self,
        l_causal: torch.Tensor,
        encoder: nn.Module,
        timestamps: torch.Tensor,
        covariates: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the total loss.

        Parameters
        ----------
        l_causal : torch.Tensor scalar
            Causal prior loss from icl_model.forward(). Already computed
            in the training loop before calling this.
        encoder : TemporalEncoder
        timestamps, covariates, padding_mask : batch tensors from collate_histories

        Returns
        -------
        total_loss : torch.Tensor scalar
        loss_components : dict
            Individual loss values for logging. Keys: 'causal', 'recon', 'consistency'.
        """
        # Skip computing terms whose weights are zero — avoids wasted compute
        # and prevents potential numerical issues in ablation runs
        if self.lambda2 > 0:
            l_recon = masked_reconstruction_loss(
                encoder=encoder,
                timestamps=timestamps,
                covariates=covariates,
                mask=padding_mask,
                mask_fraction=self.mask_fraction,
            )
        else:
            l_recon = torch.tensor(0.0, device=l_causal.device)

        if self.lambda3 > 0:
            l_consistency = temporal_consistency_loss(
                encoder=encoder,
                timestamps=timestamps,
                covariates=covariates,
                mask=padding_mask,
            )
        else:
            l_consistency = torch.tensor(0.0, device=l_causal.device)

        total = (
            self.lambda1 * l_causal
            + self.lambda2 * l_recon
            + self.lambda3 * l_consistency
        )

        components = {
            "causal": l_causal.item(),
            "recon": l_recon.item(),
            "consistency": l_consistency.item(),
            "total": total.item(),
        }

        return total, components
