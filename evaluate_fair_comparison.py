"""
evaluate_fair_comparison.py — Fair Comparison: Same Features to CausalPFN vs TabPFN-X

Feeds identical features to both estimators to isolate representation quality
from estimator quality. Evaluates all 4 conditions:
    - CausalPFN + Static features
    - TabPFN-X  + Static features
    - CausalPFN + Temporal embeddings
    - TabPFN-X  + Temporal embeddings

Usage:
    python evaluate_fair_comparison.py \
        --checkpoint_dir ./checkpoints \
        --output_dir ./results_fair \
        --device cuda
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
import argparse
import warnings
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from dgp import make_dgp, DGP_CONFIGS
from temporal_encoder import TemporalEncoder, collate_histories
from lora import inject_lora

# Setup TabPFN client authentication (token should be set via environment variable or notebook)
try:
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor
    # Token is set externally (e.g., in Colab notebook or via TABPFN_TOKEN env var)
    if os.environ.get("TABPFN_TOKEN"):
        tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
warnings.filterwarnings("ignore", category=FutureWarning)

SEEDS = [42, 123, 999]
EVAL_SEEDS = 3
TABPFN_MAX_TRAIN = 1000


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
# Feature extraction
# ------------------------------------------------------------------

def extract_mean_std_features(batch):
    X = []
    for h in batch.histories:
        X.append(np.concatenate([h.covariates.mean(0), h.covariates.std(0) + 1e-6]))
    X = np.array(X, dtype=np.float32)
    N, d = X.shape
    if d < 99:
        X = np.concatenate([X, np.zeros((N, 99 - d), dtype=np.float32)], axis=1)
    return X[:, :99]


def extract_temporal_embeddings(encoder, batch, device):
    encoder.eval()
    with torch.no_grad():
        ts, cov, mask = collate_histories(batch.histories, device)
        z = encoder(ts, cov, mask)
    return z.cpu().numpy()


# ------------------------------------------------------------------
# Encoder loading
# ------------------------------------------------------------------

def load_encoder_from_checkpoint(ckpt, device):
    """Load encoder from checkpoint."""
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
    return encoder


# ------------------------------------------------------------------
# CausalPFN and TabPFN estimators
# ------------------------------------------------------------------

def causalpfn_cate(X_ctx, t_ctx, y_ctx, X_qry, device_str, icl_model=None):
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


def tabpfn_x_learner_cate(X_ctx, t_ctx, y_ctx, X_qry,
                           TabPFNRegressor, api_version, device_arg):
    def make_reg():
        if api_version == "cloud_api":
            return TabPFNRegressor(n_estimators=4)  # Cloud API
        elif api_version == "new_api":
            return TabPFNRegressor(device=device_arg)
        return TabPFNRegressor()

    ctrl_idx = np.where(t_ctx == 0)[0]
    trt_idx  = np.where(t_ctx == 1)[0]

    if len(ctrl_idx) < 5 or len(trt_idx) < 5:
        raise ValueError(f"Too few per arm: ctrl={len(ctrl_idx)}, trt={len(trt_idx)}")

    if len(ctrl_idx) > TABPFN_MAX_TRAIN:
        ctrl_idx = np.random.choice(ctrl_idx, TABPFN_MAX_TRAIN, replace=False)
    if len(trt_idx) > TABPFN_MAX_TRAIN:
        trt_idx = np.random.choice(trt_idx, TABPFN_MAX_TRAIN, replace=False)

    mu_0 = make_reg(); mu_0.fit(X_ctx[ctrl_idx], y_ctx[ctrl_idx])
    mu_1 = make_reg(); mu_1.fit(X_ctx[trt_idx],  y_ctx[trt_idx])

    D1 = y_ctx[trt_idx]  - mu_0.predict(X_ctx[trt_idx])
    D0 = mu_1.predict(X_ctx[ctrl_idx]) - y_ctx[ctrl_idx]

    tau_1 = make_reg(); tau_1.fit(X_ctx[trt_idx],  D1)
    tau_0 = make_reg(); tau_0.fit(X_ctx[ctrl_idx], D0)

    e = float(t_ctx.mean())
    return (1 - e) * tau_1.predict(X_qry) + e * tau_0.predict(X_qry)


# ------------------------------------------------------------------
# Evaluate one config
# ------------------------------------------------------------------

def evaluate_config(
    config_name, checkpoint_dir, TabPFNRegressor, api_version, device_arg,
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
        ckpt_path = Path(checkpoint_dir) / config_name / str(train_seed) / "latest.pt"
        if not ckpt_path.exists():
            log(f"  WARNING: checkpoint not found: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt["config"]
        encoder = load_encoder_from_checkpoint(ckpt, device)

        # Fresh CausalPFN + LoRA
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

            X_s = extract_mean_std_features(batch)
            X_t = extract_temporal_embeddings(encoder, batch, device)

            for feat_name, X in [("static", X_s), ("temporal", X_t)]:
                X_ctx_f = X[ctx_idx]
                X_qry_f = X[qry_idx]

                # CausalPFN
                try:
                    icl = est_lora.icl_model if feat_name == "temporal" else None
                    cate = causalpfn_cate(X_ctx_f, t_ctx, y_ctx, X_qry_f,
                                          device_str, icl_model=icl)
                    results[f"causalpfn_{feat_name}"]["pehe"].append(
                        pehe(cate, true_cate))
                    results[f"causalpfn_{feat_name}"]["ate"].append(
                        ate_relative_error(cate.mean(), true_ate))
                except Exception as e:
                    log(f"  causalpfn_{feat_name} FAILED "
                        f"(train={train_seed}, eval={eval_seed}): {e}")

                # TabPFN-X
                try:
                    cate = tabpfn_x_learner_cate(
                        X_ctx_f, t_ctx, y_ctx, X_qry_f,
                        TabPFNRegressor, api_version, device_arg
                    )
                    results[f"tabpfn_x_{feat_name}"]["pehe"].append(
                        pehe(cate, true_cate))
                    results[f"tabpfn_x_{feat_name}"]["ate"].append(
                        ate_relative_error(cate.mean(), true_ate))
                except Exception as e:
                    log(f"  tabpfn_x_{feat_name} FAILED "
                        f"(train={train_seed}, eval={eval_seed}): {e}")

        log(
            f"  seed={train_seed} | " +
            " | ".join(
                f"{k.replace('causalpfn','cpfn').replace('tabpfn_x','tabx')}"
                f"={np.mean(v['pehe']):.4f}"
                for k, v in results.items() if v["pehe"]
            )
        )

    def agg(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": len(vals)}

    return {k: {"pehe": agg(v["pehe"]), "ate": agg(v["ate"])}
            for k, v in results.items()}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_fair_comparison(
    checkpoint_dir="./checkpoints",
    output_dir="./results_fair",
    device_str="cuda",
    n_eval_patients=1000,
    configs=None,
):
    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if configs is None:
        configs = list(DGP_CONFIGS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path     = out_dir / f"fair_log_{timestamp}.txt"
    results_path = out_dir / f"fair_results_{timestamp}.json"
    table_path   = out_dir / f"fair_table_{timestamp}.txt"

    # Load TabPFN (using cloud API via tabpfn_client)
    TabPFNReg, api_ver, dev_arg = None, None, None
    try:
        from tabpfn_client import TabPFNRegressor
        # Cloud API - no device argument needed
        TabPFNReg, api_ver, dev_arg = TabPFNRegressor, "cloud_api", None
        # Verify TabPFN works by instantiating
        _test_reg = TabPFNRegressor()
        print("TabPFN client loaded and ready to use (cloud API)")
    except Exception as e:
        print(f"ERROR: TabPFN not available: {e}")
        return

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    all_results = {}

    print("\n" + "=" * 65)
    print("PHASE 1: Setup Complete")
    print("=" * 65)
    print(f"  TabPFN: {api_ver}")
    print(f"  Device: {device}")
    print(f"  Configs to evaluate: {len(configs)}")

    with open(log_path, "w") as log_file:
        log_file.write(f"Fair Comparison — {timestamp}\n")
        log_file.write(f"Checkpoints: {checkpoint_dir}\n")
        log_file.write(f"TabPFN API: {api_ver}\n")
        log_file.write("="*65 + "\n\n")

        print("\n" + "=" * 65)
        print("PHASE 2: Evaluation")
        print("=" * 65)

        for config_name in configs:
            print(f"\n{'='*65}\nConfig: {config_name}\n{'='*65}", flush=True)
            log_file.write(f"\n{'='*65}\nConfig: {config_name}\n{'='*65}\n")

            r = evaluate_config(
                config_name, checkpoint_dir,
                TabPFNReg, api_ver, dev_arg,
                device, device_str, n_eval_patients, log_file
            )
            all_results[config_name] = r

            # Per-config summary
            log_file.write(f"\n  Summary:\n")
            for cond in ["causalpfn_static", "tabpfn_x_static",
                         "causalpfn_temporal", "tabpfn_x_temporal"]:
                p = r[cond]["pehe"]
                a = r[cond]["ate"]
                if p["mean"] is not None:
                    line = (f"  {cond:<25} | "
                            f"PEHE={p['mean']:.4f}±{p['std']:.4f} | "
                            f"ATE_err={a['mean']:.1f}%±{a['std']:.1f}%")
                else:
                    line = f"  {cond:<25} | NO RESULTS"
                print(line, flush=True)
                log_file.write(line + "\n")
            log_file.flush()

        log_file.write("\n" + "=" * 65 + "\n")
        log_file.write("PHASE 3: Saving Results\n")
        log_file.write("=" * 65 + "\n")

    print("\n" + "=" * 65)
    print("PHASE 3: Saving Results")
    print("=" * 65)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results JSON saved: {results_path}", flush=True)

    table_lines = _build_table(all_results, configs)
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines))
    print(f"  Table saved: {table_path}", flush=True)
    print(f"  Log saved: {log_path}", flush=True)

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)
    for line in table_lines:
        print(line, flush=True)

    return all_results


def _build_table(all_results, configs):
    CONDS = [
        ("causalpfn_static",   "CausalPFN+Static"),
        ("tabpfn_x_static",    "TabPFN-X+Static"),
        ("causalpfn_temporal", "CausalPFN+Temporal"),
        ("tabpfn_x_temporal",  "TabPFN-X+Temporal"),
    ]

    def cell(r, metric):
        if r is None:
            return "N/A"
        v = r.get(metric, {})
        if v.get("mean") is None:
            return "N/A"
        if metric == "ate":
            return f"{v['mean']:.1f}%±{v['std']:.1f}%"
        return f"{v['mean']:.4f}±{v['std']:.4f}"

    lines = []
    for metric, label in [("pehe", "PEHE (lower is better)"),
                           ("ate",  "ATE Relative Error % (lower is better)")]:
        lines.append("")
        lines.append("="*95)
        lines.append(f"FAIR COMPARISON — {label}")
        lines.append("="*95)
        col_w = 22
        header = f"{'Config':<22}" + "".join(f"{n:<{col_w}}" for _, n in CONDS)
        lines.append(header)
        lines.append("-"*95)

        for config in configs:
            r = all_results.get(config, {})
            row = f"{config:<22}"
            for cond_key, _ in CONDS:
                row += f"{cell(r.get(cond_key), metric):<{col_w}}"
            lines.append(row)

        lines.append("="*95)

    lines += [
        "",
        "KEY QUESTIONS:",
        "  Q1: Does CausalPFN+Temporal beat TabPFN-X+Temporal?",
        "      YES -> CausalPFN's causal prior adds value over meta-learners.",
        "      NO  -> OOD gap is more fundamental than rank and normalization.",
        "  Q2: Does temporal embedding improve over static for both estimators?",
        "      Compare +Temporal vs +Static rows for each estimator.",
    ]
    return lines


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir",    default="./checkpoints")
    parser.add_argument("--output_dir",        default="./results_fair")
    parser.add_argument("--device",            default="cuda")
    parser.add_argument("--n_eval_patients",   type=int, default=1000)
    parser.add_argument("--configs",           nargs="+",
                        default=list(DGP_CONFIGS.keys()),
                        choices=list(DGP_CONFIGS.keys()))
    args = parser.parse_args()

    run_fair_comparison(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        n_eval_patients=args.n_eval_patients,
        configs=args.configs,
    )
