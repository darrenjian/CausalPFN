"""
evaluate.py — Evaluation Across DGP Configurations

Computes PEHE and ATE relative error for:
    1. Your trained temporal encoder (Track B)
    2. Static baselines (last observation, mean+std)
    3. Darren's best static summary from Track A (plug in manually)

The main result table: as beta and eta increase, static baselines should
degrade while your encoder holds up. This is the paper's core claim.

Usage:
    # Evaluate a trained encoder checkpoint:
    python evaluate.py --checkpoint ./checkpoints/strong_temporal/latest.pt

    # Evaluate all DGP configs:
    python evaluate.py --checkpoint ./checkpoints/strong_temporal/latest.pt --all_configs

    # Baselines only (no trained encoder needed):
    python evaluate.py --baselines_only
"""

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dgp import make_dgp, DGP_CONFIGS, DGPBatch
from src.temporal_encoder import TemporalEncoder, collate_histories
from src.lora import inject_lora


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def pehe(cate_pred: np.ndarray, cate_true: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effects.
    sqrt(E[(tau_hat - tau)^2]) — lower is better.
    """
    return float(np.sqrt(np.mean((cate_pred - cate_true) ** 2)))


def ate_relative_error(ate_pred: float, ate_true: float) -> float:
    """
    |ATE_hat - ATE_true| / |ATE_true| * 100 (as percentage) — lower is better.
    """
    if abs(ate_true) < 1e-8:
        return float("nan")
    return float(abs(ate_pred - ate_true) / abs(ate_true) * 100)


# ------------------------------------------------------------------
# Static baselines
# ------------------------------------------------------------------

def make_static_features_last_obs(batch: DGPBatch) -> np.ndarray:
    """
    Baseline: use only the last observed covariate values.
    Completely ignores timestamps and temporal patterns.
    Shape: (N, n_covariates)
    """
    features = []
    for h in batch.histories:
        features.append(h.covariates[-1])  # Last observation
    return np.array(features, dtype=np.float32)


def make_static_features_mean_std(batch: DGPBatch) -> np.ndarray:
    """
    Baseline: mean and std of each covariate over the window.
    Slightly richer than last-obs but still ignores timing.
    Shape: (N, 2 * n_covariates)
    """
    features = []
    for h in batch.histories:
        mean = h.covariates.mean(axis=0)
        std = h.covariates.std(axis=0) + 1e-6
        features.append(np.concatenate([mean, std]))
    return np.array(features, dtype=np.float32)


def pad_features_to_99(X: np.ndarray) -> np.ndarray:
    """
    Pad static feature vectors to 99 dimensions with zeros.
    CausalPFN expects 99-dim input (plus treatment = 100 total).
    """
    N, d = X.shape
    if d >= 99:
        return X[:, :99]
    pad = np.zeros((N, 99 - d), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)


# ------------------------------------------------------------------
# Evaluate one method on one DGP config
# ------------------------------------------------------------------

def evaluate_method(
    X: np.ndarray,       # (N, 99) feature matrix (from encoder or baseline)
    batch: DGPBatch,
    est,                 # CATEEstimator instance (already loaded)
    n_eval: int = 500,
) -> dict:
    """
    Fit CausalPFN on context, estimate CATE/ATE on query, compare to ground truth.

    Parameters
    ----------
    X : (N, 99) feature matrix
    batch : DGPBatch with ground-truth Y0, Y1, A, Y_obs
    est : CATEEstimator — we reuse the loaded model, just call fit() again
    n_eval : int — number of query patients to evaluate on

    Returns
    -------
    dict with 'pehe', 'ate_rel_error', 'ate_pred', 'ate_true'
    """
    N = X.shape[0]
    n_context = max(10, int(N * 0.7))
    perm = np.random.permutation(N)
    ctx_idx = perm[:n_context]
    qry_idx = perm[n_context:n_context + n_eval]

    X_ctx = X[ctx_idx]
    t_ctx = batch.A[ctx_idx]
    y_ctx = batch.Y_obs[ctx_idx]

    X_qry = X[qry_idx]
    true_cate = batch.true_CATE[qry_idx]
    true_ate = float(batch.true_CATE.mean())

    # Fit CausalPFN on context embeddings
    est.X_train = X_ctx
    est.t_train = t_ctx
    est.y_train = y_ctx
    est._train_weak_learner(X_ctx, t_ctx, y_ctx)
    est.prediction_temperature = 1.0

    # Estimate CATE on query
    cate_pred = est.estimate_cate(X_qry)
    ate_pred = float(cate_pred.mean())

    return {
        "pehe": pehe(cate_pred, true_cate),
        "ate_rel_error": ate_relative_error(ate_pred, true_ate),
        "ate_pred": ate_pred,
        "ate_true": true_ate,
    }


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str = None,
    config_names: list = None,
    n_eval_patients: int = 1000,
    n_seeds: int = 3,
    device_str: str = "cuda",
    baselines_only: bool = False,
) -> dict:
    """
    Evaluate all methods across specified DGP configurations.

    Returns a results dict structured as:
        results[config_name][method_name] = {pehe, ate_rel_error, ...}
    """
    if config_names is None:
        config_names = list(DGP_CONFIGS.keys())

    # Fix random seed for reproducibility across runs
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load CausalPFN
    from causalpfn import CATEEstimator
    est = CATEEstimator(device=device_str)
    est.load_model()

    # Load encoder + LoRA if checkpoint provided
    encoder = None
    if checkpoint_path is not None and not baselines_only:
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["config"]

        encoder = TemporalEncoder(
            n_covariates=cfg["n_covariates"],
            d_pe=16,
            d_model=cfg["d_model"],
            n_heads=4,
            n_layers=cfg["n_layers"],
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        # Inject LoRA and load LoRA weights
        # Checkpoint saves LoRA weights under key "lora" (not "icl_model")
        inject_lora(est.icl_model, rank=cfg["lora_rank"])
        est.icl_model = est.icl_model.to(device)  # move LoRA params to GPU first
        lora_weights = ckpt["lora"]
        state = est.icl_model.state_dict()
        state.update(lora_weights)
        est.icl_model.load_state_dict(state, strict=False)

        print(f"Loaded encoder + LoRA from: {checkpoint_path}")

    results = {}

    for config_name in config_names:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")
        results[config_name] = {}

        # Aggregate over multiple seeds for stability
        method_results = {
            "last_obs": [],
            "mean_std": [],
        }
        if encoder is not None:
            method_results["temporal_encoder"] = []

        for seed in range(n_seeds):
            dgp = make_dgp(config_name, n_covariates=5, seed=seed + 100)
            batch = dgp.sample_batch(
                n_patients=n_eval_patients, window_hours=48.0
            )

            # Baseline: last observation
            X_last = pad_features_to_99(make_static_features_last_obs(batch))
            r = evaluate_method(X_last, batch, est)
            method_results["last_obs"].append(r)

            # Baseline: mean + std
            X_ms = pad_features_to_99(make_static_features_mean_std(batch))
            r = evaluate_method(X_ms, batch, est)
            method_results["mean_std"].append(r)

            # Temporal encoder
            if encoder is not None:
                with torch.no_grad():
                    timestamps, covariates, padding_mask = collate_histories(
                        batch.histories, device
                    )
                    z = encoder(timestamps, covariates, padding_mask)
                    X_enc = z.cpu().numpy()
                r = evaluate_method(X_enc, batch, est)
                method_results["temporal_encoder"].append(r)

        # Aggregate seed results
        for method, seed_results in method_results.items():
            agg = {
                "pehe_mean": np.mean([r["pehe"] for r in seed_results]),
                "pehe_std": np.std([r["pehe"] for r in seed_results]),
                "ate_rel_error_mean": np.mean([r["ate_rel_error"] for r in seed_results]),
                "ate_rel_error_std": np.std([r["ate_rel_error"] for r in seed_results]),
            }
            results[config_name][method] = agg

            print(
                f"  {method:<22} | "
                f"PEHE={agg['pehe_mean']:.4f}±{agg['pehe_std']:.4f} | "
                f"ATE_err={agg['ate_rel_error_mean']:.1f}%±{agg['ate_rel_error_std']:.1f}%"
            )

    return results


# ------------------------------------------------------------------
# Results table printer
# ------------------------------------------------------------------

def print_results_table(results: dict) -> None:
    """
    Print the main results table for the paper.
    Rows = DGP configurations, Columns = methods.
    """
    configs = list(results.keys())
    methods = list(results[configs[0]].keys())

    print("\n" + "="*80)
    print("MAIN RESULTS TABLE — PEHE (lower is better)")
    print("="*80)
    header = f"{'Config':<25}" + "".join(f"{m:<22}" for m in methods)
    print(header)
    print("-"*80)
    for config in configs:
        row = f"{config:<25}"
        for method in methods:
            r = results[config][method]
            row += f"{r['pehe_mean']:.4f}±{r['pehe_std']:.4f}      "
        print(row)

    print("\n" + "="*80)
    print("MAIN RESULTS TABLE — ATE RELATIVE ERROR % (lower is better)")
    print("="*80)
    print(header)
    print("-"*80)
    for config in configs:
        row = f"{config:<25}"
        for method in methods:
            r = results[config][method]
            row += f"{r['ate_rel_error_mean']:.1f}%±{r['ate_rel_error_std']:.1f}%         "
        print(row)
    print("="*80)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained encoder checkpoint")
    parser.add_argument("--all_configs", action="store_true",
                        help="Evaluate on all 4 DGP configurations")
    parser.add_argument("--config", default="strong_temporal",
                        choices=list(DGP_CONFIGS.keys()))
    parser.add_argument("--n_eval_patients", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baselines_only", action="store_true")
    args = parser.parse_args()

    config_names = list(DGP_CONFIGS.keys()) if args.all_configs else [args.config]

    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        config_names=config_names,
        n_eval_patients=args.n_eval_patients,
        n_seeds=args.n_seeds,
        device_str=args.device,
        baselines_only=args.baselines_only,
    )

    print_results_table(results)

    # Save results
    out_path = Path("./results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
