"""
evaluate_tabpfn.py — TabPFN + Meta-Learner Baselines

Evaluates two TabPFN-based causal estimation strategies:

    TabPFN-TLearner:
        Train one TabPFN per treatment arm. CATE = mu_1(x) - mu_0(x).
        Simple and fast. Ignores treatment propensity.

    TabPFN-XLearner:
        Stage 1: Fit mu_0, mu_1 with TabPFN (one per arm).
        Stage 2: Impute pseudo-outcomes:
                 D_i^1 = Y_i - mu_0(X_i)  for treated
                 D_i^0 = mu_1(X_i) - Y_i  for control
        Stage 3: Fit CATE models tau_0, tau_1 on pseudo-outcomes.
        Stage 4: Weight by propensity: CATE = g(x)*tau_0(x) + (1-g(x))*tau_1(x)
        More robust under treatment imbalance.

Both methods use the same static features as our baselines (mean+std of
covariates), padded to match TabPFN's expected input format.

TabPFN constraints (enforced with guards):
    - Max 1000 training samples (context set)
    - Max 100 features
    - Classification only in original API — we use regression mode

All outputs are saved to files. Nothing important is terminal-only.

Usage:
    pip install tabpfn
    python evaluate_tabpfn.py --output_dir ./results_tabpfn --device cuda
"""

import json
import argparse
import warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from dgp import make_dgp, DGP_CONFIGS

# Suppress TabPFN's internal warnings that clutter output
warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------------
# TabPFN version check and import
# ------------------------------------------------------------------

def load_tabpfn_regressor(device_str):
    """
    Load TabPFN regressor, handling API differences across versions.
    Returns a regressor class and a flag for which API is being used.
    """
    try:
        from tabpfn import TabPFNRegressor
        # Test instantiation to confirm regression mode works
        reg = TabPFNRegressor(device=device_str)
        return TabPFNRegressor, "new_api", device_str
    except ImportError:
        raise ImportError(
            "TabPFN not found. Install with: pip install tabpfn\n"
            "If already installed, check version: pip show tabpfn"
        )
    except Exception as e:
        # Some versions require different instantiation
        try:
            from tabpfn import TabPFNRegressor
            reg = TabPFNRegressor()
            return TabPFNRegressor, "no_device_api", None
        except Exception as e2:
            raise RuntimeError(
                f"TabPFN failed to initialize: {e}\n"
                f"Fallback also failed: {e2}\n"
                f"Try: pip install tabpfn --upgrade"
            )


def make_tabpfn_regressor(TabPFNRegressor, api_version, device_str):
    """Create a fresh TabPFN regressor instance."""
    if api_version == "new_api":
        return TabPFNRegressor(device=device_str)
    else:
        return TabPFNRegressor()


# ------------------------------------------------------------------
# Feature extraction (same static features as other baselines)
# ------------------------------------------------------------------

TABPFN_MAX_TRAIN = 1000   # TabPFN hard limit
TABPFN_MAX_FEATURES = 100  # TabPFN hard limit

def extract_static_features(batch, n_features=10):
    """
    Extract mean+std static features from irregular histories.
    Shape: (N, 2 * n_covariates) — same as mean_std baseline.
    Padded/truncated to n_features for TabPFN compatibility.
    """
    features = []
    for h in batch.histories:
        mean = h.covariates.mean(axis=0)
        std  = h.covariates.std(axis=0) + 1e-6
        features.append(np.concatenate([mean, std]))
    X = np.array(features, dtype=np.float32)

    # Enforce TabPFN feature limit
    if X.shape[1] > TABPFN_MAX_FEATURES:
        X = X[:, :TABPFN_MAX_FEATURES]

    return X


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def pehe(cate_pred, cate_true):
    return float(np.sqrt(np.mean((cate_pred - cate_true) ** 2)))

def ate_relative_error(ate_pred, ate_true):
    if abs(ate_true) < 1e-8:
        return float("nan")
    return float(abs(ate_pred - ate_true) / abs(ate_true) * 100)


# ------------------------------------------------------------------
# T-Learner
# ------------------------------------------------------------------

def t_learner_cate(
    X_train, t_train, y_train,
    X_test,
    TabPFNRegressor, api_version, device_str,
):
    """
    T-Learner: fit one model per treatment arm, predict CATE as difference.

    mu_0: fitted on control patients (t=0)
    mu_1: fitted on treated patients (t=1)
    CATE(x) = mu_1(x) - mu_0(x)
    """
    ctrl_idx = np.where(t_train == 0)[0]
    trt_idx  = np.where(t_train == 1)[0]

    # Guard: need at least a few samples per arm
    if len(ctrl_idx) < 5 or len(trt_idx) < 5:
        raise ValueError(
            f"Too few samples per arm: control={len(ctrl_idx)}, "
            f"treated={len(trt_idx)}. Check treatment positivity."
        )

    # Enforce TabPFN training limit
    if len(ctrl_idx) > TABPFN_MAX_TRAIN:
        ctrl_idx = np.random.choice(ctrl_idx, TABPFN_MAX_TRAIN, replace=False)
    if len(trt_idx) > TABPFN_MAX_TRAIN:
        trt_idx = np.random.choice(trt_idx, TABPFN_MAX_TRAIN, replace=False)

    # Fit control model
    mu_0 = make_tabpfn_regressor(TabPFNRegressor, api_version, device_str)
    mu_0.fit(X_train[ctrl_idx], y_train[ctrl_idx])

    # Fit treatment model
    mu_1 = make_tabpfn_regressor(TabPFNRegressor, api_version, device_str)
    mu_1.fit(X_train[trt_idx], y_train[trt_idx])

    # Predict potential outcomes on test set
    mu_0_pred = mu_0.predict(X_test)
    mu_1_pred = mu_1.predict(X_test)

    return mu_1_pred - mu_0_pred


# ------------------------------------------------------------------
# X-Learner
# ------------------------------------------------------------------

def x_learner_cate(
    X_train, t_train, y_train,
    X_test,
    TabPFNRegressor, api_version, device_str,
):
    """
    X-Learner (Kunzel et al. 2019):

    Stage 1: Fit outcome models mu_0, mu_1 per arm.
    Stage 2: Impute pseudo-outcomes on the opposite arm's data.
    Stage 3: Fit CATE models tau_0, tau_1 on pseudo-outcomes.
    Stage 4: Combine with propensity weighting.

    More robust than T-Learner under treatment imbalance.
    """
    ctrl_idx = np.where(t_train == 0)[0]
    trt_idx  = np.where(t_train == 1)[0]

    if len(ctrl_idx) < 5 or len(trt_idx) < 5:
        raise ValueError(
            f"Too few samples per arm: control={len(ctrl_idx)}, "
            f"treated={len(trt_idx)}."
        )

    # Subsample if needed for TabPFN limit
    ctrl_use = ctrl_idx if len(ctrl_idx) <= TABPFN_MAX_TRAIN else \
        np.random.choice(ctrl_idx, TABPFN_MAX_TRAIN, replace=False)
    trt_use  = trt_idx  if len(trt_idx)  <= TABPFN_MAX_TRAIN else \
        np.random.choice(trt_idx, TABPFN_MAX_TRAIN, replace=False)

    # --- Stage 1: Outcome models ---
    mu_0 = make_tabpfn_regressor(TabPFNRegressor, api_version, device_str)
    mu_0.fit(X_train[ctrl_use], y_train[ctrl_use])

    mu_1 = make_tabpfn_regressor(TabPFNRegressor, api_version, device_str)
    mu_1.fit(X_train[trt_use], y_train[trt_use])

    # --- Stage 2: Pseudo-outcomes ---
    # For treated: D^1 = Y_i - mu_0(X_i)  (what treatment added)
    mu_0_on_trt = mu_0.predict(X_train[trt_use])
    D1 = y_train[trt_use] - mu_0_on_trt

    # For control: D^0 = mu_1(X_i) - Y_i  (what treatment would have added)
    mu_1_on_ctrl = mu_1.predict(X_train[ctrl_use])
    D0 = mu_1_on_ctrl - y_train[ctrl_use]

    # --- Stage 3: CATE models on pseudo-outcomes ---
    tau_1 = make_tabpfn_regressor(TabPFNRegressor, api_version, device_str)
    tau_1.fit(X_train[trt_use], D1)

    tau_0 = make_tabpfn_regressor(TabPFNRegressor, api_version, device_str)
    tau_0.fit(X_train[ctrl_use], D0)

    # --- Stage 4: Propensity-weighted combination ---
    # Estimate propensity e(x) = P(T=1|X) using a simple logistic proxy.
    # TabPFN is a regressor here; use mean treatment rate as propensity approximation
    # for simplicity (avoids a separate classifier fit).
    propensity = t_train.mean()  # scalar — marginal propensity
    # Constant propensity gives: CATE = (1-e)*tau_1 + e*tau_0
    tau_1_pred = tau_1.predict(X_test)
    tau_0_pred = tau_0.predict(X_test)

    cate = (1 - propensity) * tau_1_pred + propensity * tau_0_pred
    return cate


# ------------------------------------------------------------------
# Evaluate one configuration
# ------------------------------------------------------------------

def evaluate_config(
    config_name,
    TabPFNRegressor,
    api_version,
    device_str,
    n_eval_patients=1000,
    n_seeds=3,
    log_file=None,
):
    """
    Evaluate TabPFN T-Learner and X-Learner on one DGP configuration.
    Returns dict with aggregated results. Logs to file if provided.
    """
    t_pehe_list, t_ate_list = [], []
    x_pehe_list, x_ate_list = [], []

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        dgp = make_dgp(config_name, n_covariates=5, seed=seed + 100)
        batch = dgp.sample_batch(n_patients=n_eval_patients, window_hours=48.0)

        X = extract_static_features(batch)
        A = batch.A
        Y_obs = batch.Y_obs
        true_cate = batch.true_CATE
        true_ate  = float(batch.true_CATE.mean())

        N = len(X)
        n_ctx = min(int(N * 0.7), TABPFN_MAX_TRAIN)
        perm = np.random.permutation(N)
        ctx_idx = perm[:n_ctx]
        qry_idx = perm[n_ctx:]

        X_ctx, t_ctx, y_ctx = X[ctx_idx], A[ctx_idx], Y_obs[ctx_idx]
        X_qry = X[qry_idx]
        true_cate_qry = true_cate[qry_idx]

        # T-Learner
        try:
            t_cate = t_learner_cate(
                X_ctx, t_ctx, y_ctx, X_qry,
                TabPFNRegressor, api_version, device_str
            )
            t_pehe_list.append(pehe(t_cate, true_cate_qry))
            t_ate_list.append(ate_relative_error(t_cate.mean(), true_ate))
            msg = f"  [seed={seed}] T-Learner: PEHE={t_pehe_list[-1]:.4f}, ATE_err={t_ate_list[-1]:.1f}%"
        except Exception as e:
            msg = f"  [seed={seed}] T-Learner FAILED: {e}"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

        # X-Learner
        try:
            x_cate = x_learner_cate(
                X_ctx, t_ctx, y_ctx, X_qry,
                TabPFNRegressor, api_version, device_str
            )
            x_pehe_list.append(pehe(x_cate, true_cate_qry))
            x_ate_list.append(ate_relative_error(x_cate.mean(), true_ate))
            msg = f"  [seed={seed}] X-Learner: PEHE={x_pehe_list[-1]:.4f}, ATE_err={x_ate_list[-1]:.1f}%"
        except Exception as e:
            msg = f"  [seed={seed}] X-Learner FAILED: {e}"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    def agg(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    return {
        "t_learner": {
            "pehe": agg(t_pehe_list),
            "ate_rel_error": agg(t_ate_list),
        },
        "x_learner": {
            "pehe": agg(x_pehe_list),
            "ate_rel_error": agg(x_ate_list),
        },
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_tabpfn_evaluation(
    output_dir="./results_tabpfn",
    device_str="cuda",
    n_eval_patients=1000,
    n_seeds=3,
    configs=None,
):
    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if configs is None:
        configs = list(DGP_CONFIGS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path    = out_dir / f"tabpfn_eval_log_{timestamp}.txt"
    results_path = out_dir / f"tabpfn_results_{timestamp}.json"
    table_path   = out_dir / f"tabpfn_table_{timestamp}.txt"

    # Check TabPFN version and load
    print("Loading TabPFN...", flush=True)
    try:
        TabPFNRegressor, api_version, device_arg = load_tabpfn_regressor(device_str)
        print(f"TabPFN loaded. API version: {api_version}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}")
        with open(log_path, "w") as f:
            f.write(f"TabPFN load failed: {e}\n")
        return

    all_results = {}

    with open(log_path, "w") as log_file:
        log_file.write(f"TabPFN Evaluation — {timestamp}\n")
        log_file.write(f"device={device_str}, n_eval_patients={n_eval_patients}, "
                       f"n_seeds={n_seeds}\n")
        log_file.write("="*65 + "\n\n")

        for config_name in configs:
            header = f"Config: {config_name}"
            print(f"\n{'='*65}")
            print(header, flush=True)
            print(f"{'='*65}")
            log_file.write(f"\n{'='*65}\n{header}\n{'='*65}\n")
            log_file.flush()

            results = evaluate_config(
                config_name=config_name,
                TabPFNRegressor=TabPFNRegressor,
                api_version=api_version,
                device_str=device_arg,
                n_eval_patients=n_eval_patients,
                n_seeds=n_seeds,
                log_file=log_file,
            )
            all_results[config_name] = results

            # Write per-config summary to log
            for method in ["t_learner", "x_learner"]:
                r = results[method]
                p = r["pehe"]
                a = r["ate_rel_error"]
                if p["mean"] is not None:
                    summary = (
                        f"  {method:<15} | "
                        f"PEHE={p['mean']:.4f}\u00b1{p['std']:.4f} | "
                        f"ATE_err={a['mean']:.1f}%\u00b1{a['std']:.1f}% "
                        f"[{p['n']} seeds]"
                    )
                else:
                    summary = f"  {method:<15} | FAILED"
                print(summary)
                log_file.write(summary + "\n")
            log_file.flush()

    # Save results JSON
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON saved: {results_path}")

    # Build and save formatted comparison table
    table_lines = _build_comparison_table(all_results, configs)
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines))
    print(f"Formatted table saved: {table_path}")

    # Also print table to terminal for immediate inspection
    for line in table_lines:
        print(line)

    return all_results


def _build_comparison_table(all_results, configs):
    """
    Build a formatted comparison table including TabPFN results.
    Also includes the static baseline numbers from the main experiment
    for reference (hardcoded from final_results.json).
    """
    # Hardcoded from your main experiment final results
    # (aggregated across 3 training seeds)
    MAIN_RESULTS = {
        "tabular_sufficient": {
            "last_obs":  {"pehe": "0.4647\u00b10.0012", "ate": "40.8%\u00b10.8%"},
            "mean_std":  {"pehe": "0.4607\u00b10.0018", "ate": "25.8%\u00b10.5%"},
            "temp_enc":  {"pehe": "0.4279\u00b10.0010", "ate": "6.8%\u00b11.4%"},
        },
        "weak_temporal": {
            "last_obs":  {"pehe": "0.4592\u00b10.0017", "ate": "37.4%\u00b11.9%"},
            "mean_std":  {"pehe": "0.4687\u00b10.0021", "ate": "30.7%\u00b11.8%"},
            "temp_enc":  {"pehe": "0.4296\u00b10.0010", "ate": "9.3%\u00b11.6%"},
        },
        "strong_temporal": {
            "last_obs":  {"pehe": "0.6379\u00b10.0246", "ate": "97.3%\u00b16.3%"},
            "mean_std":  {"pehe": "0.5733\u00b10.0086", "ate": "69.3%\u00b14.2%"},
            "temp_enc":  {"pehe": "0.4491\u00b10.0056", "ate": "26.8%\u00b13.8%"},
        },
        "asymmetric": {
            "last_obs":  {"pehe": "0.4851\u00b10.0091", "ate": "50.5%\u00b13.7%"},
            "mean_std":  {"pehe": "0.4647\u00b10.0163", "ate": "27.2%\u00b110.1%"},
            "temp_enc":  {"pehe": "0.4340\u00b10.0026", "ate": "14.6%\u00b13.7%"},
        },
    }

    lines = []
    lines.append("="*95)
    lines.append("FULL COMPARISON TABLE — PEHE (lower is better)")
    lines.append("="*95)
    header = (f"{'Config':<22} {'Last Obs':>14} {'Mean+Std':>14} "
              f"{'TabPFN-T':>14} {'TabPFN-X':>14} {'Temp.Enc (ours)':>18}")
    lines.append(header)
    lines.append("-"*95)

    for config in configs:
        main = MAIN_RESULTS.get(config, {})
        tab  = all_results.get(config, {})
        t_p = tab.get("t_learner", {}).get("pehe", {})
        x_p = tab.get("x_learner", {}).get("pehe", {})

        t_str = f"{t_p['mean']:.4f}\u00b1{t_p['std']:.4f}" if t_p.get("mean") else "N/A"
        x_str = f"{x_p['mean']:.4f}\u00b1{x_p['std']:.4f}" if x_p.get("mean") else "N/A"

        lines.append(
            f"{config:<22} "
            f"{main.get('last_obs', {}).get('pehe', 'N/A'):>14} "
            f"{main.get('mean_std', {}).get('pehe', 'N/A'):>14} "
            f"{t_str:>14} "
            f"{x_str:>14} "
            f"{main.get('temp_enc', {}).get('pehe', 'N/A'):>18}"
        )

    lines.append("")
    lines.append("="*95)
    lines.append("FULL COMPARISON TABLE — ATE RELATIVE ERROR % (lower is better)")
    lines.append("="*95)
    lines.append(header)
    lines.append("-"*95)

    for config in configs:
        main = MAIN_RESULTS.get(config, {})
        tab  = all_results.get(config, {})
        t_a = tab.get("t_learner", {}).get("ate_rel_error", {})
        x_a = tab.get("x_learner", {}).get("ate_rel_error", {})

        t_str = f"{t_a['mean']:.1f}%\u00b1{t_a['std']:.1f}%" if t_a.get("mean") else "N/A"
        x_str = f"{x_a['mean']:.1f}%\u00b1{x_a['std']:.1f}%" if x_a.get("mean") else "N/A"

        lines.append(
            f"{config:<22} "
            f"{main.get('last_obs', {}).get('ate', 'N/A'):>14} "
            f"{main.get('mean_std', {}).get('ate', 'N/A'):>14} "
            f"{t_str:>14} "
            f"{x_str:>14} "
            f"{main.get('temp_enc', {}).get('ate', 'N/A'):>18}"
        )

    lines.append("="*95)
    lines.append("")
    lines.append("Note: Last Obs, Mean+Std, Temp.Enc results are from main experiment")
    lines.append("(aggregated across 3 training seeds). TabPFN results use static")
    lines.append("mean+std features — same input as Mean+Std baseline.")
    return lines


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TabPFN T-Learner and X-Learner baselines"
    )
    parser.add_argument("--output_dir",  default="./results_tabpfn",
                        help="Directory for all output files")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--n_eval_patients", type=int, default=1000)
    parser.add_argument("--n_seeds",     type=int, default=3)
    parser.add_argument("--configs",     nargs="+",
                        default=list(DGP_CONFIGS.keys()),
                        choices=list(DGP_CONFIGS.keys()),
                        help="Which DGP configs to evaluate")
    args = parser.parse_args()

    run_tabpfn_evaluation(
        output_dir=args.output_dir,
        device_str=args.device,
        n_eval_patients=args.n_eval_patients,
        n_seeds=args.n_seeds,
        configs=args.configs,
    )
