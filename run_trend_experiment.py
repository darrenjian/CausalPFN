"""
run_trend_experiment.py — Train + Evaluate on Trend-Confounded DGP Configs

Self-contained script that:
    1. Trains temporal encoders on the two new trend configs
    2. Runs the fair comparison (all 4 conditions) on both configs
    3. Saves all results to files

This directly answers whether temporal representation adds value when
the confounder is a *directional trend* rather than observation frequency —
a confounder that is genuinely invisible to mean+std summaries.

Usage:
    python run_trend_experiment.py \
        --output_dir ./results_trend \
        --device cuda
"""

import os
import json
import math
import argparse
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
warnings.filterwarnings("ignore", category=FutureWarning)

# Import from existing project files
from dgp import make_dgp, DGP_CONFIGS
from temporal_encoder import TemporalEncoder, collate_histories
from lora import inject_lora
from losses import JointLoss

TREND_CONFIGS = ["strong_temporal_trend", "trend_only"]
SEEDS = [42, 123, 999]
EVAL_SEEDS = 3
TABPFN_MAX_TRAIN = 1000


# ------------------------------------------------------------------
# Autograd-safe patch (same as train.py — required for backward pass)
# ------------------------------------------------------------------

def apply_causalpfn_patch(icl_model):
    """
    Patch clip_outliers and normalize_data to be autograd-safe.
    Must be applied after icl_model.to(device).
    """
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


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_one_run(
    config_name, seed, checkpoint_dir, device, device_str,
    n_steps=3000, batch_size=64,
    lambda1=1.0, lambda2=0.5, lambda3=0.1,
    n_covariates=5, window_hours=48.0,
    log_file=None,
):
    """Train encoder + LoRA on one config/seed. Save checkpoint."""
    def log(msg):
        print(msg, flush=True)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pt"

    # Resume if checkpoint exists
    if latest_ckpt.exists():
        log(f"  Checkpoint exists at {latest_ckpt} — skipping training.")
        return

    dgp = make_dgp(config_name, n_covariates=n_covariates, seed=seed)
    log(f"  DGP: {config_name} beta={dgp.beta} eta={dgp.eta} "
        f"gamma_trend={dgp.gamma_trend} eta_trend={dgp.eta_trend}")

    encoder = TemporalEncoder(
        n_covariates=n_covariates, d_pe=16, d_model=64,
        n_heads=4, n_layers=2, dropout=0.1,
    ).to(device)

    from causalpfn import CATEEstimator
    est = CATEEstimator(device=device_str)
    est.load_model()
    icl_model = est.icl_model

    inject_lora(icl_model, rank=8, alpha=8.0)
    icl_model = icl_model.to(device)
    apply_causalpfn_patch(icl_model)

    trainable = (
        [p for p in encoder.parameters() if p.requires_grad]
        + [p for p in icl_model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)

    warmup = max(1, n_steps // 20)
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    joint_loss = JointLoss(lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    t0 = time.time()
    for step in range(n_steps):
        encoder.train()
        icl_model.train()
        optimizer.zero_grad()

        batch = dgp.sample_batch(n_patients=batch_size, window_hours=window_hours)
        timestamps, covariates, padding_mask = collate_histories(batch.histories, device)
        z = encoder(timestamps, covariates, padding_mask)

        A     = torch.from_numpy(batch.A).float().to(device)
        Y0    = torch.from_numpy(batch.Y0).float().to(device)
        Y1    = torch.from_numpy(batch.Y1).float().to(device)
        Y_obs = torch.from_numpy(batch.Y_obs).float().to(device)

        N = z.shape[0]
        n_ctx = max(4, int(N * 0.7))
        perm = torch.randperm(N, device=device)
        ctx_idx, qry_idx = perm[:n_ctx], perm[n_ctx:]

        if len(qry_idx) < 2:
            qry_idx = perm[-(max(2, N // 5)):]
            ctx_idx = perm[:N - len(qry_idx)]

        l_causal = icl_model(
            X_context=z[ctx_idx].unsqueeze(0).clone(),
            t_context=A[ctx_idx].unsqueeze(0).clone(),
            y_context=Y_obs[ctx_idx].unsqueeze(0).clone(),
            X_query=z[qry_idx].unsqueeze(0).clone(),
            E_y0_query=Y0[qry_idx].unsqueeze(0).clone(),
            E_y1_query=Y1[qry_idx].unsqueeze(0).clone(),
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
        nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log(
                f"  step={step:4d} | total={components['total']:.4f} | "
                f"causal={components['causal']:.4f} | "
                f"recon={components['recon']:.4f} | {elapsed:.0f}s"
            )

    # Save checkpoint
    lora_weights = {
        k: v for k, v in icl_model.state_dict().items()
        if 'lora_A' in k or 'lora_B' in k
    }
    ckpt = {
        "step": n_steps - 1,
        "config_name": config_name,
        "seed": seed,
        "encoder": encoder.state_dict(),
        "lora": lora_weights,
        "config": {
            "config_name": config_name,
            "n_covariates": n_covariates,
            "d_pe": 16, "d_model": 64,
            "n_heads": 4, "n_layers": 2,
            "lora_rank": 8, "lora_alpha": 8.0,
        },
    }
    torch.save(ckpt, latest_ckpt)
    log(f"  Saved: {latest_ckpt}")


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def pehe(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def ate_err(pred_ate, true_ate):
    if abs(true_ate) < 1e-8:
        return float("nan")
    return float(abs(pred_ate - true_ate) / abs(true_ate) * 100)


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def mean_std_features(batch):
    X = []
    for h in batch.histories:
        X.append(np.concatenate([h.covariates.mean(0), h.covariates.std(0) + 1e-6]))
    X = np.array(X, dtype=np.float32)
    N, d = X.shape
    if d < 99:
        X = np.concatenate([X, np.zeros((N, 99 - d), dtype=np.float32)], axis=1)
    return X[:, :99]


def temporal_embeddings(encoder, batch, device):
    encoder.eval()
    with torch.no_grad():
        ts, cov, mask = collate_histories(batch.histories, device)
        z = encoder(ts, cov, mask)
    return z.cpu().numpy()


# ------------------------------------------------------------------
# CausalPFN estimator
# ------------------------------------------------------------------

def causalpfn_estimate(X_ctx, t_ctx, y_ctx, X_qry, device_str, icl_model=None):
    from causalpfn import CATEEstimator
    est = CATEEstimator(device=device_str)
    est.load_model()
    if icl_model is not None:
        est.icl_model = icl_model
    est.X_train = X_ctx
    est.t_train = t_ctx
    est.y_train = y_ctx
    est._train_weak_learner(X_ctx, t_ctx, y_ctx)
    est.prediction_temperature = 1.0
    return est.estimate_cate(X_qry)


# ------------------------------------------------------------------
# TabPFN X-Learner
# ------------------------------------------------------------------

def tabpfn_x_estimate(X_ctx, t_ctx, y_ctx, X_qry, TabPFNReg, api_ver, dev_arg):
    def reg():
        return TabPFNReg(device=dev_arg) if api_ver == "new_api" else TabPFNReg()

    ctrl = np.where(t_ctx == 0)[0]
    trt  = np.where(t_ctx == 1)[0]
    if len(ctrl) < 5 or len(trt) < 5:
        raise ValueError(f"Too few per arm: ctrl={len(ctrl)}, trt={len(trt)}")

    if len(ctrl) > TABPFN_MAX_TRAIN:
        ctrl = np.random.choice(ctrl, TABPFN_MAX_TRAIN, replace=False)
    if len(trt) > TABPFN_MAX_TRAIN:
        trt = np.random.choice(trt, TABPFN_MAX_TRAIN, replace=False)

    mu_0 = reg(); mu_0.fit(X_ctx[ctrl], y_ctx[ctrl])
    mu_1 = reg(); mu_1.fit(X_ctx[trt],  y_ctx[trt])

    D1 = y_ctx[trt]  - mu_0.predict(X_ctx[trt])
    D0 = mu_1.predict(X_ctx[ctrl]) - y_ctx[ctrl]

    tau_1 = reg(); tau_1.fit(X_ctx[trt],  D1)
    tau_0 = reg(); tau_0.fit(X_ctx[ctrl], D0)

    e = float(t_ctx.mean())
    return (1 - e) * tau_1.predict(X_qry) + e * tau_0.predict(X_qry)


# ------------------------------------------------------------------
# Fair comparison evaluation
# ------------------------------------------------------------------

def evaluate_fair(
    config_name, checkpoint_base, TabPFNReg, api_ver, dev_arg,
    device, device_str, n_eval_patients=1000, log_file=None,
):
    def log(msg):
        print(msg, flush=True)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    results = {
        "causalpfn_static":   {"pehe": [], "ate": []},
        "tabpfn_x_static":    {"pehe": [], "ate": []},
        "causalpfn_temporal": {"pehe": [], "ate": []},
        "tabpfn_x_temporal":  {"pehe": [], "ate": []},
    }

    for train_seed in SEEDS:
        ckpt_path = Path(checkpoint_base) / str(train_seed) / "latest.pt"
        if not ckpt_path.exists():
            log(f"  WARNING: missing {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt["config"]

        encoder = TemporalEncoder(
            n_covariates=cfg["n_covariates"], d_pe=16,
            d_model=cfg["d_model"], n_heads=4, n_layers=cfg["n_layers"],
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        from causalpfn import CATEEstimator
        est_lora = CATEEstimator(device=device_str)
        est_lora.load_model()
        inject_lora(est_lora.icl_model, rank=cfg["lora_rank"])
        est_lora.icl_model = est_lora.icl_model.to(device)
        state = est_lora.icl_model.state_dict()
        state.update(ckpt["lora"])
        est_lora.icl_model.load_state_dict(state, strict=False)
        est_lora.icl_model.eval()

        for eval_seed in range(EVAL_SEEDS):
            torch.manual_seed(eval_seed)
            np.random.seed(eval_seed)

            dgp   = make_dgp(config_name, n_covariates=5, seed=eval_seed + 100)
            batch = dgp.sample_batch(n_patients=n_eval_patients, window_hours=48.0)

            N = n_eval_patients
            n_ctx = min(int(N * 0.7), TABPFN_MAX_TRAIN)
            perm = np.random.permutation(N)
            ctx_idx, qry_idx = perm[:n_ctx], perm[n_ctx:]

            true_cate = batch.true_CATE[qry_idx]
            true_ate  = float(batch.true_CATE.mean())
            t_ctx = batch.A[ctx_idx]
            y_ctx = batch.Y_obs[ctx_idx]

            X_s = mean_std_features(batch)
            X_t = temporal_embeddings(encoder, batch, device)

            for feat_name, X in [("static", X_s), ("temporal", X_t)]:
                X_ctx_f = X[ctx_idx]
                X_qry_f = X[qry_idx]

                for method_name, fn, extra in [
                    ("causalpfn",
                     causalpfn_estimate,
                     {"device_str": device_str,
                      "icl_model": est_lora.icl_model if feat_name == "temporal" else None}),
                    ("tabpfn_x",
                     tabpfn_x_estimate,
                     {"TabPFNReg": TabPFNReg, "api_ver": api_ver, "dev_arg": dev_arg}),
                ]:
                    key = f"{method_name}_{feat_name}"
                    try:
                        if method_name == "causalpfn":
                            cate_pred = fn(X_ctx_f, t_ctx, y_ctx, X_qry_f, **extra)
                        else:
                            cate_pred = fn(X_ctx_f, t_ctx, y_ctx, X_qry_f, **extra)
                        results[key]["pehe"].append(pehe(cate_pred, true_cate))
                        results[key]["ate"].append(ate_err(cate_pred.mean(), true_ate))
                    except Exception as e:
                        log(f"  {key} FAILED (train={train_seed}, eval={eval_seed}): {e}")

        log(
            f"  seed={train_seed} done | "
            + " | ".join(
                f"{k}={np.mean(v['pehe']):.4f}"
                for k, v in results.items() if v["pehe"]
            )
        )

    def agg(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    return {
        k: {"pehe": agg(v["pehe"]), "ate": agg(v["ate"])}
        for k, v in results.items()
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(output_dir="./results_trend", device_str="cuda", n_eval_patients=1000):
    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path     = out_dir / f"trend_log_{timestamp}.txt"
    results_path = out_dir / f"trend_results_{timestamp}.json"
    table_path   = out_dir / f"trend_table_{timestamp}.txt"

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load TabPFN
    try:
        from tabpfn import TabPFNRegressor
        _ = TabPFNRegressor(device=device_str)
        TabPFNReg, api_ver, dev_arg = TabPFNRegressor, "new_api", device_str
        print("TabPFN loaded.", flush=True)
    except Exception:
        try:
            from tabpfn import TabPFNRegressor
            TabPFNReg, api_ver, dev_arg = TabPFNRegressor, "no_device_api", None
        except Exception as e:
            print(f"ERROR: TabPFN failed: {e}")
            return

    all_results = {}

    with open(log_path, "w") as log_file:
        log_file.write(f"Trend Experiment — {timestamp}\n")
        log_file.write("="*65 + "\n\n")

        # --- Phase 1: Training ---
        log_file.write("PHASE 1: TRAINING\n")
        log_file.write("="*65 + "\n")
        print("\n" + "="*65)
        print("PHASE 1: TRAINING")
        print("="*65, flush=True)

        for config_name in TREND_CONFIGS:
            for seed in SEEDS:
                msg = f"\nTraining config={config_name} seed={seed}"
                print(msg, flush=True)
                log_file.write(msg + "\n")

                ckpt_dir = ckpt_root / config_name / str(seed)
                train_one_run(
                    config_name=config_name,
                    seed=seed,
                    checkpoint_dir=ckpt_dir,
                    device=device,
                    device_str=device_str,
                    n_steps=3000,
                    batch_size=64,
                    log_file=log_file,
                )

        # --- Phase 2: Evaluation ---
        log_file.write("\n\nPHASE 2: FAIR COMPARISON EVALUATION\n")
        log_file.write("="*65 + "\n")
        print("\n" + "="*65)
        print("PHASE 2: FAIR COMPARISON EVALUATION")
        print("="*65, flush=True)

        for config_name in TREND_CONFIGS:
            msg = f"\nConfig: {config_name}"
            print(msg, flush=True)
            log_file.write(msg + "\n")

            config_results = evaluate_fair(
                config_name=config_name,
                checkpoint_base=ckpt_root / config_name,
                TabPFNReg=TabPFNReg,
                api_ver=api_ver,
                dev_arg=dev_arg,
                device=device,
                device_str=device_str,
                n_eval_patients=n_eval_patients,
                log_file=log_file,
            )
            all_results[config_name] = config_results

            # Per-config summary
            log_file.write(f"\n  Summary:\n")
            CONDS = ["causalpfn_static", "tabpfn_x_static",
                     "causalpfn_temporal", "tabpfn_x_temporal"]
            for cond in CONDS:
                r = config_results[cond]
                p, a = r["pehe"], r["ate"]
                if p["mean"] is not None:
                    line = (
                        f"  {cond:<25} | "
                        f"PEHE={p['mean']:.4f}\u00b1{p['std']:.4f} | "
                        f"ATE_err={a['mean']:.1f}%\u00b1{a['std']:.1f}%"
                    )
                else:
                    line = f"  {cond:<25} | NO RESULTS"
                print(line, flush=True)
                log_file.write(line + "\n")

    # Save JSON
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON: {results_path}", flush=True)

    # Build table
    table_lines = _build_table(all_results)
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines))
    print(f"Table: {table_path}", flush=True)
    for line in table_lines:
        print(line, flush=True)


def _build_table(all_results):
    CONDS = [
        ("causalpfn_static",   "CausalPFN+Static"),
        ("tabpfn_x_static",    "TabPFN-X+Static"),
        ("causalpfn_temporal", "CausalPFN+Temporal"),
        ("tabpfn_x_temporal",  "TabPFN-X+Temporal"),
    ]

    def cell(r, metric):
        v = r.get(metric, {})
        if v.get("mean") is None:
            return "N/A"
        if metric == "ate":
            return f"{v['mean']:.1f}%\u00b1{v['std']:.1f}%"
        return f"{v['mean']:.4f}\u00b1{v['std']:.4f}"

    lines = ["", "="*90,
             "TREND EXPERIMENT — FAIR COMPARISON TABLE",
             "Configs use recency trend as confounder — invisible to mean+std features",
             "="*90]

    for metric, label in [("pehe", "PEHE (lower is better)"),
                           ("ate",  "ATE Relative Error % (lower is better)")]:
        lines += ["", f"--- {label} ---"]
        col_w = 22
        lines.append(f"{'Config':<25}" + "".join(f"{n:<{col_w}}" for _, n in CONDS))
        lines.append("-"*90)
        for config in TREND_CONFIGS:
            row = f"{config:<25}"
            for cond_key, _ in CONDS:
                r = all_results.get(config, {}).get(cond_key, {})
                row += f"{cell(r, metric):<{col_w}}"
            lines.append(row)

    lines += [
        "", "="*90,
        "KEY QUESTIONS:",
        "  Q1: Does TabPFN-X+Temporal beat TabPFN-X+Static?",
        "      YES -> Trend representation adds value even to a general-purpose learner.",
        "      NO  -> The encoder is not capturing the trend signal effectively.",
        "  Q2: How does TabPFN-X+Static compare to original strong_temporal results?",
        "      If TabPFN-X+Static degrades significantly vs original strong_temporal,",
        "      the trend confounder is genuinely harder to handle without temporal features.",
        "  Q3: Does CausalPFN+Temporal vs CausalPFN+Static gap widen vs original results?",
        "      YES -> The encoder is more useful when the confounder is trend-based.",
        "="*90,
    ]
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results_trend")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_eval_patients", type=int, default=1000)
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        device_str=args.device,
        n_eval_patients=args.n_eval_patients,
    )
