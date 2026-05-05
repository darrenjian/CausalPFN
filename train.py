"""
train.py — Training Loop for Temporal Encoder + LoRA Fine-Tuning

Trains the temporal encoder E_phi jointly with LoRA adapters in the
CausalPFN backbone using a synthetic DGP that generates batches with
known ground-truth potential outcomes.

Training procedure per batch:
    1. Sample a batch of synthetic patients from the DGP
    2. Run encoder: H_i -> z_i (shape: batch x 99)
    3. Split into context and query sets
    4. Call icl_model.forward() with z embeddings to get L_causal
    5. Compute L_reconstruction and L_consistency via losses.py
    6. Backpropagate L_total through encoder and LoRA adapters only
    7. Log and checkpoint

Usage:
    # Smoke test
    python train.py --config strong_temporal --n_steps 5 --batch_size 32

    # Full training run
    python train.py --config strong_temporal --n_steps 3000 \
        --batch_size 64 --checkpoint_dir ./checkpoints/strong_temporal/42 \
        --seed 42 --device cuda
"""

import os
import math
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dgp import make_dgp, DGP_CONFIGS, DGPBatch
from temporal_encoder import TemporalEncoder, collate_histories
from lora import inject_lora, print_parameter_summary
from losses import JointLoss


# ------------------------------------------------------------------
# Context / query split
# ------------------------------------------------------------------

def split_context_query(z, A, Y_obs, Y0, Y1, context_fraction=0.7):
    N = z.shape[0]
    n_context = max(4, int(N * context_fraction))
    perm = torch.randperm(N, device=z.device)
    ctx_idx = perm[:n_context]
    qry_idx = perm[n_context:]

    if len(qry_idx) < 2:
        qry_idx = perm[-(max(2, N // 5)):]
        ctx_idx = perm[:N - len(qry_idx)]

    X_context = z[ctx_idx].unsqueeze(0)
    t_context = A[ctx_idx].unsqueeze(0)
    y_context = Y_obs[ctx_idx].unsqueeze(0)
    X_query   = z[qry_idx].unsqueeze(0)
    E_y0      = Y0[qry_idx].unsqueeze(0)
    E_y1      = Y1[qry_idx].unsqueeze(0)

    return X_context, t_context, y_context, X_query, E_y0, E_y1


# ------------------------------------------------------------------
# Autograd-safe patch
# ------------------------------------------------------------------

def apply_causalpfn_patch():
    import causalpfn.models.model as _cpfn_model

    def _safe_maskmean(x, mask, dim):
        x_safe = torch.where(mask, x.clone(), torch.zeros_like(x))
        denom = mask.sum(dim=dim, keepdim=True).clamp(min=1)
        return x_safe.sum(dim=dim, keepdim=True) / denom

    def _safe_maskstd(x, mask, dim=0):
        num = mask.sum(dim=dim, keepdim=True).clamp(min=1)
        mean = _safe_maskmean(x, mask, dim=dim)
        diffs = torch.where(mask, (x - mean).clone(), torch.zeros_like(x))
        return ((diffs ** 2).sum(dim=0, keepdim=True) / (num - 1).clamp(min=1)) ** 0.5

    def _safe_normalize_data(data, eval_pos):
        X = data[:eval_pos].clone() if eval_pos > 0 else data.clone()
        mask = ~torch.isnan(X)
        mean = _safe_maskmean(X, mask, dim=0)
        std = _safe_maskstd(X, mask, dim=0) + 1e-6
        return (data - mean) / std

    def _safe_clip_outliers(data, eval_pos, n_sigma=4):
        assert len(data.shape) == 3
        X = data[:eval_pos].clone() if eval_pos > 0 else data.clone()
        mask = ~torch.isnan(X)
        mean = _safe_maskmean(X, mask, dim=0)
        cutoff = n_sigma * _safe_maskstd(X, mask, dim=0)
        return torch.clamp(data, mean - cutoff, mean + cutoff)

    _cpfn_model.maskmean = _safe_maskmean
    _cpfn_model.maskstd = _safe_maskstd
    _cpfn_model.normalize_data = _safe_normalize_data
    _cpfn_model.clip_outliers = _safe_clip_outliers
    print("Autograd-safe patch applied.")


# ------------------------------------------------------------------
# One training step
# ------------------------------------------------------------------

def train_step(encoder, icl_model, batch, joint_loss, optimizer, device,
               context_fraction=0.7):
    encoder.train()
    icl_model.train()
    optimizer.zero_grad()

    timestamps, covariates, padding_mask = collate_histories(batch.histories, device)
    z = encoder(timestamps, covariates, padding_mask)

    A     = torch.from_numpy(batch.A).float().to(device)
    Y0    = torch.from_numpy(batch.Y0).float().to(device)
    Y1    = torch.from_numpy(batch.Y1).float().to(device)
    Y_obs = torch.from_numpy(batch.Y_obs).float().to(device)

    X_ctx, t_ctx, y_ctx, X_qry, E_y0, E_y1 = split_context_query(
        z, A, Y_obs, Y0, Y1, context_fraction
    )

    l_causal = icl_model(
        X_context=X_ctx.clone(),
        t_context=t_ctx.clone(),
        y_context=y_ctx.clone(),
        X_query=X_qry.clone(),
        E_y0_query=E_y0.clone(),
        E_y1_query=E_y1.clone(),
    )
    if l_causal.dim() > 0:
        l_causal = l_causal.mean()

    total_loss, components = joint_loss(
        l_causal=l_causal,
        encoder=encoder,
        timestamps=timestamps,
        covariates=covariates,
        padding_mask=padding_mask,
    )

    total_loss.backward()
    trainable = (
        [p for p in encoder.parameters() if p.requires_grad]
        + [p for p in icl_model.parameters() if p.requires_grad]
    )
    nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    optimizer.step()
    return components


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train(
    config_name: str = "strong_temporal",
    n_steps: int = 3000,
    batch_size: int = 64,
    n_covariates: int = 5,
    window_hours: float = 48.0,
    context_fraction: float = 0.7,
    d_pe: int = 16,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    lora_rank: int = 32,
    lora_alpha: float = 32.0,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    lambda3: float = 0.1,
    mask_fraction: float = 0.15,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_every: int = 500,
    log_every: int = 50,
    seed: int = 42,
    device_str: str = "cuda",
):
    """
    Full training loop with checkpointing and LR schedule.
    Checkpoints saved to checkpoint_dir/latest.pt.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"LoRA rank: {lora_rank}")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dgp = make_dgp(config_name, n_covariates=n_covariates, seed=seed)
    print(f"DGP config: {config_name} (beta={dgp.beta}, eta={dgp.eta})")

    encoder = TemporalEncoder(
        n_covariates=n_covariates,
        d_pe=d_pe,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    from causalpfn import CATEEstimator
    est = CATEEstimator(device=device_str)
    est.load_model()
    icl_model = est.icl_model

    inject_lora(icl_model, rank=lora_rank, alpha=lora_alpha)
    icl_model = icl_model.to(device)
    apply_causalpfn_patch()
    print_parameter_summary(icl_model, encoder)

    trainable_params = (
        [p for p in encoder.parameters() if p.requires_grad]
        + [p for p in icl_model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay
    )

    warmup_steps = max(1, n_steps // 20)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    joint_loss = JointLoss(
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        mask_fraction=mask_fraction,
    )

    # Resume from checkpoint if available
    start_step = 0
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        lora_state = ckpt.get("lora", {})
        if lora_state:
            state = icl_model.state_dict()
            state.update(lora_state)
            icl_model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    log_history = []
    t0 = time.time()
    print(f"\nStarting training: {n_steps} steps, batch_size={batch_size}")
    print("-" * 65)

    for step in range(start_step, n_steps):
        batch = dgp.sample_batch(n_patients=batch_size, window_hours=window_hours)
        components = train_step(
            encoder, icl_model, batch, joint_loss, optimizer, device,
            context_fraction
        )
        scheduler.step()

        if step % log_every == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            entry = {"step": step, "elapsed_s": round(elapsed, 1),
                     "lr": round(lr_now, 6),
                     **{k: round(v, 4) for k, v in components.items()}}
            log_history.append(entry)
            print(
                f"Step {step:5d} | total={components['total']:.4f} | "
                f"causal={components['causal']:.4f} | "
                f"recon={components['recon']:.4f} | "
                f"consistency={components['consistency']:.4f} | "
                f"lr={lr_now:.2e} | {elapsed:.0f}s"
            )

        if (step + 1) % checkpoint_every == 0 or step == n_steps - 1:
            lora_weights = {
                k: v for k, v in icl_model.state_dict().items()
                if 'lora_A' in k or 'lora_B' in k
            }
            ckpt_data = {
                "step": step,
                "config_name": config_name,
                "seed": seed,
                "encoder": encoder.state_dict(),
                "lora": lora_weights,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": {
                    "config_name": config_name,
                    "n_covariates": n_covariates,
                    "d_pe": d_pe,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "encoder_version": "v2",
                },
            }
            torch.save(ckpt_data, latest_ckpt)
            torch.save(ckpt_data, ckpt_dir / f"step_{step:05d}.pt")
            print(f"  -> Checkpoint saved: {latest_ckpt}")

    with open(ckpt_dir / "train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nTraining complete. Checkpoints in: {ckpt_dir}")
    return encoder, icl_model


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train temporal encoder + LoRA"
    )
    parser.add_argument("--config", default="strong_temporal",
                        choices=list(DGP_CONFIGS.keys()))
    parser.add_argument("--n_steps",      type=int,   default=3000)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--n_covariates", type=int,   default=5)
    parser.add_argument("--d_model",      type=int,   default=64)
    parser.add_argument("--n_layers",     type=int,   default=2)
    parser.add_argument("--lora_rank",    type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--lambda1",      type=float, default=1.0)
    parser.add_argument("--lambda2",      type=float, default=0.5)
    parser.add_argument("--lambda3",      type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--log_every",    type=int,   default=50)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    train(
        config_name=args.config,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_covariates=args.n_covariates,
        d_model=args.d_model,
        n_layers=args.n_layers,
        lora_rank=args.lora_rank,
        lr=args.lr,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        seed=args.seed,
        device_str=args.device,
    )
