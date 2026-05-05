"""
evaluate_mimic.py — Semi-Synthetic MIMIC-IV Evaluation

Evaluates the temporal encoder on real MIMIC-IV trajectories with simulated
outcomes. Produces the same comparison table as evaluate_fair_comparison.py:
    - CausalPFN + Static features
    - TabPFN-X  + Static features
    - CausalPFN + Temporal embeddings
    - TabPFN-X  + Temporal embeddings

The key question: does the encoder trained on synthetic OU-process data
produce useful representations for real clinical observation patterns?

Usage:
    # With a trained encoder checkpoint
    python evaluate_mimic.py \
        --checkpoint_dir ./checkpoints \
        --mimic_dir ./mimic_iv \
        --output_dir ./results_mimic \
        --device cpu

    # Baselines only (no encoder needed)
    python evaluate_mimic.py \
        --baselines_only \
        --mimic_dir ./mimic_iv \
        --output_dir ./results_mimic
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

from mimic_cohort import extract_cohort
from mimic_dgp import make_mimic_dgp, MIMIC_DGP_CONFIGS
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

TRAIN_SEEDS = [42, 123, 999]
SCM_SEEDS = 5
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
    cfg = ckpt["config"]

    encoder = TemporalEncoder(
        n_covariates=cfg["n_covariates"], d_pe=16,
        d_model=cfg["d_model"], n_heads=4, n_layers=cfg["n_layers"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder


# ------------------------------------------------------------------
# CausalPFN and TabPFN estimators
# ------------------------------------------------------------------

_cached_cpfn_est = None

def _get_cpfn_estimator(device_str):
    global _cached_cpfn_est
    if _cached_cpfn_est is None:
        from causalpfn import CATEEstimator
        est = CATEEstimator(device=device_str)
        est.load_model()
        _cached_cpfn_est = est
    return _cached_cpfn_est

def causalpfn_cate(X_ctx, t_ctx, y_ctx, X_qry, device_str, icl_model=None):
    import copy
    from sklearn.ensemble import GradientBoostingRegressor
    est = copy.copy(_get_cpfn_estimator(device_str))
    if icl_model is not None:
        est.icl_model = icl_model
    est.X_train = X_ctx
    est.t_train = t_ctx
    est.y_train = y_ctx
    # Patch: min_samples_leaf=0 when n < 100 crashes sklearn >=1.8
    orig_train = est._train_weak_learner.__func__
    def patched_train(self, X, t, y):
        from sklearn.preprocessing import OneHotEncoder
        self.t_transformer = OneHotEncoder(sparse_output=False, categories="auto", drop="first")
        T = self.t_transformer.fit_transform(t.reshape(-1, 1))
        self._d_t = (T.shape[1],)
        feat_arr = np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)
        self.stratifier = GradientBoostingRegressor(
            n_estimators=100, max_depth=6,
            min_samples_leaf=max(1, int(X.shape[0] / 100)),
            random_state=111,
        )
        self.stratifier.fit(feat_arr, y)
    import types
    est._train_weak_learner = types.MethodType(patched_train, est)
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
    trt_idx = np.where(t_ctx == 1)[0]
    if len(ctrl_idx) < 5 or len(trt_idx) < 5:
        raise ValueError(f"Too few per arm: ctrl={len(ctrl_idx)}, trt={len(trt_idx)}")

    if len(ctrl_idx) > TABPFN_MAX_TRAIN:
        ctrl_idx = np.random.choice(ctrl_idx, TABPFN_MAX_TRAIN, replace=False)
    if len(trt_idx) > TABPFN_MAX_TRAIN:
        trt_idx = np.random.choice(trt_idx, TABPFN_MAX_TRAIN, replace=False)

    mu_0 = make_reg(); mu_0.fit(X_ctx[ctrl_idx], y_ctx[ctrl_idx])
    mu_1 = make_reg(); mu_1.fit(X_ctx[trt_idx], y_ctx[trt_idx])

    D1 = y_ctx[trt_idx] - mu_0.predict(X_ctx[trt_idx])
    D0 = mu_1.predict(X_ctx[ctrl_idx]) - y_ctx[ctrl_idx]

    tau_1 = make_reg(); tau_1.fit(X_ctx[trt_idx], D1)
    tau_0 = make_reg(); tau_0.fit(X_ctx[ctrl_idx], D0)

    e = float(t_ctx.mean())
    return (1 - e) * tau_1.predict(X_qry) + e * tau_0.predict(X_qry)


# ------------------------------------------------------------------
# Distribution comparison
# ------------------------------------------------------------------

def compare_distributions(cohort, batch):
    """Compare MIMIC temporal feature distributions to synthetic DGP."""
    from dgp import make_dgp

    synth_dgp = make_dgp('strong_temporal', n_covariates=5, seed=42)
    synth_batch = synth_dgp.sample_batch(n_patients=300, window_hours=48.0)

    lines = []
    lines.append("\nTemporal Feature Distribution Comparison: MIMIC vs Synthetic")
    lines.append("-" * 65)

    for name, mimic_vals, synth_vals in [
        ("f_freq", batch.f_freq, synth_batch.f_freq),
        ("trend", batch.trend, synth_batch.trend),
    ]:
        lines.append(f"  {name}:")
        lines.append(f"    MIMIC:     mean={mimic_vals.mean():.3f}, "
                     f"std={mimic_vals.std():.3f}, "
                     f"range=[{mimic_vals.min():.3f}, {mimic_vals.max():.3f}]")
        lines.append(f"    Synthetic: mean={synth_vals.mean():.3f}, "
                     f"std={synth_vals.std():.3f}, "
                     f"range=[{synth_vals.min():.3f}, {synth_vals.max():.3f}]")

    obs_mimic = [h.n_obs for h in batch.histories]
    obs_synth = [h.n_obs for h in synth_batch.histories]
    lines.append(f"  n_obs:")
    lines.append(f"    MIMIC:     mean={np.mean(obs_mimic):.1f}, "
                 f"range=[{min(obs_mimic)}, {max(obs_mimic)}]")
    lines.append(f"    Synthetic: mean={np.mean(obs_synth):.1f}, "
                 f"range=[{min(obs_synth)}, {max(obs_synth)}]")

    corr_mean = float(np.corrcoef(batch.trend,
        np.array([h.covariates.mean() for h in batch.histories]))[0, 1])
    corr_std = float(np.corrcoef(batch.trend,
        np.array([h.covariates.std() for h in batch.histories]))[0, 1])
    lines.append(f"  MIMIC trend invisibility check:")
    lines.append(f"    corr(trend, mean): {corr_mean:.3f}")
    lines.append(f"    corr(trend, std):  {corr_std:.3f}")
    lines.append("-" * 65)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------

def evaluate_mimic(
    checkpoint_dir="./checkpoints",
    mimic_dir="./mimic_iv",
    output_dir="./results_mimic",
    device_str="cpu",
    n_eval_patients=500,
    configs=None,
    baselines_only=False,
):
    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if configs is None:
        configs = list(MIMIC_DGP_CONFIGS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"mimic_eval_log_{timestamp}.txt"
    results_path = out_dir / f"mimic_eval_results_{timestamp}.json"
    table_path = out_dir / f"mimic_eval_table_{timestamp}.txt"

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

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
        print(f"WARNING: TabPFN not available: {e}")

    # Pre-load CausalPFN model (cached for all subsequent calls)
    print("Loading CausalPFN model...", flush=True)
    _get_cpfn_estimator(device_str)
    print("CausalPFN model loaded.", flush=True)

    # Extract MIMIC cohort
    print("\n" + "=" * 65)
    print("PHASE 1: MIMIC-IV Cohort Extraction")
    print("=" * 65)
    cohort = extract_cohort(mimic_dir, n_covariates=5)

    all_results = {}

    with open(log_path, "w") as log_file:
        def log(msg):
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        log(f"Semi-Synthetic MIMIC Evaluation — {timestamp}")
        log(f"Cohort: {len(cohort.histories)} patients, "
            f"{cohort.feature_names}")
        log(f"Baselines only: {baselines_only}")
        log("=" * 65)

        # Distribution comparison
        test_dgp = make_mimic_dgp('strong_temporal', cohort, seed=42)
        test_batch = test_dgp.generate_batch(n_patients=300)
        dist_report = compare_distributions(cohort, test_batch)
        log(dist_report)

        # Load encoder checkpoints
        log(f"\n{'='*65}")
        log("PHASE 2: Loading Encoder Checkpoints")
        log("=" * 65)
        encoders = {}
        if not baselines_only:
            for train_seed in TRAIN_SEEDS:
                ckpt_path = None
                for config_name in configs:
                    candidate = (Path(checkpoint_dir) / config_name /
                                 str(train_seed) / "latest.pt")
                    if candidate.exists():
                        ckpt_path = candidate
                        break
                if ckpt_path is None:
                    # Try flat structure
                    candidate = (Path(checkpoint_dir) / str(train_seed) /
                                 "latest.pt")
                    if candidate.exists():
                        ckpt_path = candidate
                if ckpt_path is None:
                    log(f"  WARNING: no checkpoint for seed {train_seed}")
                    continue

                ckpt = torch.load(ckpt_path, map_location=device,
                                  weights_only=False)
                encoder = load_encoder_from_checkpoint(ckpt, device)
                cfg = ckpt["config"]

                # Load LoRA
                from causalpfn import CATEEstimator
                est_lora = CATEEstimator(device=device_str)
                est_lora.load_model()
                inject_lora(est_lora.icl_model,
                            rank=cfg.get("lora_rank", 8))
                est_lora.icl_model = est_lora.icl_model.to(device)
                state = est_lora.icl_model.state_dict()
                state.update(ckpt["lora"])
                est_lora.icl_model.load_state_dict(state, strict=False)
                est_lora.icl_model.eval()

                encoders[train_seed] = {
                    "encoder": encoder,
                    "icl_model": est_lora.icl_model,
                }
                log(f"  Loaded encoder seed={train_seed} from {ckpt_path}")

        if baselines_only:
            log("  Skipping encoder loading (baselines_only=True)")
        elif not encoders:
            log("  WARNING: No encoder checkpoints found!")
        else:
            log(f"  Successfully loaded {len(encoders)} encoder(s)")

        # Evaluate each config
        log(f"\n{'='*65}")
        log("PHASE 3: Evaluation")
        log(f"Configs to evaluate: {len(configs)} | SCM seeds per config: {SCM_SEEDS}")
        log(f"TabPFN status: {'loaded' if TabPFNReg else 'NOT available'}")
        log(f"Encoders loaded: {len(encoders)} (seeds: {list(encoders.keys()) if encoders else 'none'})")
        log("=" * 65)

        for config_idx, config_name in enumerate(configs):
            log(f"\n[Config {config_idx+1}/{len(configs)}] {config_name}")
            log("-" * 40)

            results = {
                "causalpfn_static":   {"pehe": [], "ate": []},
                "tabpfn_x_static":    {"pehe": [], "ate": []},
            }
            if encoders:
                results["causalpfn_temporal"] = {"pehe": [], "ate": []}
                results["tabpfn_x_temporal"] = {"pehe": [], "ate": []}

            for scm_seed in range(SCM_SEEDS):
                log(f"  [SCM seed {scm_seed+1}/{SCM_SEEDS}] Generating batch...", )
                dgp = make_mimic_dgp(config_name, cohort,
                                     seed=scm_seed + 200)
                batch = dgp.generate_batch(n_patients=n_eval_patients)
                log(f"  [SCM seed {scm_seed+1}/{SCM_SEEDS}] Batch ready, running methods...")

                N = n_eval_patients
                n_ctx = min(int(N * 0.7), TABPFN_MAX_TRAIN)
                perm = np.random.permutation(N)
                ctx_idx, qry_idx = perm[:n_ctx], perm[n_ctx:]

                true_cate = batch.true_CATE[qry_idx]
                true_ate = float(batch.true_CATE.mean())
                t_ctx = batch.A[ctx_idx]
                y_ctx = batch.Y_obs[ctx_idx]

                X_s = extract_mean_std_features(batch)

                # Static baselines
                for method_name, fn in [
                    ("causalpfn", lambda xc, tc, yc, xq:
                        causalpfn_cate(xc, tc, yc, xq, device_str)),
                    ("tabpfn_x", lambda xc, tc, yc, xq:
                        tabpfn_x_learner_cate(
                            xc, tc, yc, xq,
                            TabPFNReg, api_ver, dev_arg)
                        if TabPFNReg else None),
                ]:
                    key = f"{method_name}_static"
                    log(f"    Running {key}...")
                    try:
                        cate = fn(X_s[ctx_idx], t_ctx, y_ctx, X_s[qry_idx])
                        if cate is not None:
                            p = pehe(cate, true_cate)
                            a = ate_relative_error(cate.mean(), true_ate)
                            results[key]["pehe"].append(p)
                            results[key]["ate"].append(a)
                            log(f"    {key} done: PEHE={p:.4f}, ATE_err={a:.1f}%")
                    except Exception as e:
                        log(f"  {key} FAILED (scm_seed={scm_seed}): {e}")

                # Temporal embeddings (one per encoder seed)
                if encoders:
                    for train_seed, enc_data in encoders.items():
                        log(f"    Running temporal methods (encoder seed={train_seed})...")
                        encoder = enc_data["encoder"]
                        icl_model = enc_data["icl_model"]

                        X_t = extract_temporal_embeddings(
                            encoder, batch, device)

                        # CausalPFN + Temporal
                        log("    Running causalpfn_temporal...")
                        try:
                            cate = causalpfn_cate(
                                X_t[ctx_idx], t_ctx, y_ctx,
                                X_t[qry_idx], device_str,
                                icl_model=icl_model)
                            p = pehe(cate, true_cate)
                            a = ate_relative_error(cate.mean(), true_ate)
                            results["causalpfn_temporal"]["pehe"].append(p)
                            results["causalpfn_temporal"]["ate"].append(a)
                            log(f"    causalpfn_temporal done: PEHE={p:.4f}, ATE_err={a:.1f}%")
                        except Exception as e:
                            log(f"  causalpfn_temporal FAILED "
                                f"(scm={scm_seed}, train={train_seed}): {e}")

                        # TabPFN-X + Temporal
                        if TabPFNReg:
                            log("    Running tabpfn_x_temporal...")
                            try:
                                cate = tabpfn_x_learner_cate(
                                    X_t[ctx_idx], t_ctx, y_ctx,
                                    X_t[qry_idx],
                                    TabPFNReg, api_ver, dev_arg)
                                p = pehe(cate, true_cate)
                                a = ate_relative_error(cate.mean(), true_ate)
                                results["tabpfn_x_temporal"]["pehe"].append(p)
                                results["tabpfn_x_temporal"]["ate"].append(a)
                                log(f"    tabpfn_x_temporal done: PEHE={p:.4f}, ATE_err={a:.1f}%")
                            except Exception as e:
                                log(f"  tabpfn_x_temporal FAILED "
                                    f"(scm={scm_seed}, train={train_seed}): "
                                    f"{e}")

                log(f"  [SCM seed {scm_seed+1}/{SCM_SEEDS}] Complete!")

            # Aggregate
            def agg(vals):
                if not vals:
                    return {"mean": None, "std": None, "n": 0}
                return {"mean": float(np.mean(vals)),
                        "std": float(np.std(vals)), "n": len(vals)}

            config_results = {
                k: {"pehe": agg(v["pehe"]), "ate": agg(v["ate"])}
                for k, v in results.items()
            }
            all_results[config_name] = config_results

            for cond, r in config_results.items():
                p, a = r["pehe"], r["ate"]
                if p["mean"] is not None:
                    log(f"  {cond:<25} | "
                        f"PEHE={p['mean']:.4f}±{p['std']:.4f} | "
                        f"ATE_err={a['mean']:.1f}%±{a['std']:.1f}%")
                else:
                    log(f"  {cond:<25} | NO RESULTS")

        log(f"\n{'='*65}")
        log("PHASE 4: Saving Results")
        log("=" * 65)

    # Save results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results JSON saved: {results_path}")

    # Build and save table
    table_lines = _build_table(all_results, configs)
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines))
    print(f"Table saved: {table_path}")
    print(f"Log saved: {log_path}")
    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)
    for line in table_lines:
        print(line)

    return all_results


def _build_table(all_results, configs):
    CONDS = [
        ("causalpfn_static",   "CPFN+Static"),
        ("tabpfn_x_static",    "TabX+Static"),
        ("causalpfn_temporal", "CPFN+Temporal"),
        ("tabpfn_x_temporal",  "TabX+Temporal"),
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

    lines = [
        "",
        "=" * 95,
        "SEMI-SYNTHETIC MIMIC-IV EVALUATION",
        "Real clinical trajectories + simulated outcomes from known SCM",
        "=" * 95,
    ]

    for metric, label in [("pehe", "PEHE (lower is better)"),
                           ("ate", "ATE Relative Error % (lower is better)")]:
        lines.append(f"\n--- {label} ---")
        col_w = 22
        header = f"{'Config':<25}" + "".join(
            f"{n:<{col_w}}" for _, n in CONDS)
        lines.append(header)
        lines.append("-" * 95)
        for config in configs:
            row = f"{config:<25}"
            r = all_results.get(config, {})
            for cond_key, _ in CONDS:
                row += f"{cell(r.get(cond_key), metric):<{col_w}}"
            lines.append(row)

    lines += [
        "",
        "=" * 95,
        "KEY QUESTIONS:",
        "  Q1: Does temporal > static on MIMIC data (same as synthetic)?",
        "      YES -> Encoder representations transfer to real clinical data.",
        "  Q2: Is the gap larger or smaller than on synthetic data?",
        "      Smaller -> Distribution shift weakens encoder representations.",
        "      Larger  -> Real observation patterns are MORE informative.",
        "  Q3: Does trend confounding (invisible to mean+std) remain hard?",
        "      Check trend_only and strong_temporal_trend configs.",
        "=" * 95,
    ]
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--mimic_dir", default="./mimic_iv")
    parser.add_argument("--output_dir", default="./results_mimic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_eval_patients", type=int, default=500)
    parser.add_argument("--configs", nargs="+", default=None,
                        choices=list(MIMIC_DGP_CONFIGS.keys()))
    parser.add_argument("--baselines_only", action="store_true")
    args = parser.parse_args()

    evaluate_mimic(
        checkpoint_dir=args.checkpoint_dir,
        mimic_dir=args.mimic_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        n_eval_patients=args.n_eval_patients,
        configs=args.configs,
        baselines_only=args.baselines_only,
    )
