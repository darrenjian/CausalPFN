# Temporal-CausalPFN

Adapting CausalPFN for causal inference on irregularly-sampled ICU data via LoRA and continuous-time positional encodings.

**Authors:** Maitri Patel (mp7075@nyu.edu), Darren Jian (dj2565@nyu.edu)

## Overview

Standard causal inference models expect fixed-length tabular inputs. ICU patients are monitored at irregular intervals, and observation frequency is itself a confounder — sicker patients get checked more often. This project builds a temporal encoder that maps variable-length patient histories to fixed embeddings and adapts the frozen CausalPFN backbone via LoRA to use them.

The pipeline has four layers:

```
Irregular history H_i ──> Temporal Encoder E_phi ──> z_i (99-dim) ──> CausalPFN+LoRA ──> CATE estimate
    (timestamps +              (transformer +              (prepend treatment       (in-context
     covariates)             continuous-time PE)             indicator t_i)          causal inference)
```

## Project Structure

```
Temporal-CausalPFN/
├── src/                          # Core model components
│   ├── temporal_encoder.py       # Transformer with continuous-time PE
│   ├── lora.py                   # LoRA injection for CausalPFN
│   └── losses.py                 # Causal + reconstruction + consistency losses
├── data/                         # Data generation
│   ├── dgp.py                    # Synthetic DGP (6 confounding configs)
│   ├── mimic_cohort.py           # MIMIC-IV cohort extraction
│   └── mimic_dgp.py              # Semi-synthetic SCM for MIMIC
├── scripts/                      # Training and evaluation
│   ├── train.py                  # Joint encoder + LoRA training
│   ├── evaluate.py               # Encoder vs static baselines
│   ├── evaluate_fair_comparison.py  # CausalPFN vs TabPFN-X
│   ├── evaluate_mimic.py         # Semi-synthetic MIMIC evaluation
│   ├── evaluate_tabpfn.py        # TabPFN baselines
│   └── run_trend_experiment.py   # Trend confounding ablations
├── notebooks/
│   └── colab_train_and_eval.ipynb  # Full pipeline on Colab GPU
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/darrenjian/Temporal-CausalPFN.git
cd Temporal-CausalPFN
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `causalpfn`, `tabpfn-client`

## Synthetic Data Generating Process

The DGP simulates ICU patients with fully known counterfactuals:

1. **Patient baseline** — latent health state `v_i ~ Uniform(0,1)` drives covariate baselines
2. **Observation times** — competing-event model: `gap = min(routine_clock, alarm_clock)`, where alarm rate increases with patient severity
3. **Covariate dynamics** — Ornstein-Uhlenbeck process mean-reverting toward patient baseline
4. **Temporal confounders** — observation frequency `f_freq = n_obs / W` and recency trend (mean slope of covariates in last 6 hours)
5. **Treatment/outcomes** — logistic treatment model and linear outcome model, both confounded by `f_freq` and trend

Six configurations control confounding strength:

| Config | beta | eta | gamma_trend | eta_trend | What it tests |
|---|---|---|---|---|---|
| `tabular_sufficient` | 0 | 0 | 0 | 0 | No temporal confounding |
| `weak_temporal` | 0.5 | 0.5 | 0 | 0 | Mild frequency confounding |
| `strong_temporal` | 2.0 | 2.0 | 0 | 0 | Heavy frequency confounding |
| `asymmetric` | 2.0 | 0 | 0 | 0 | Frequency confounds treatment only |
| `strong_temporal_trend` | 2.0 | 2.0 | 2.0 | 2.0 | Both frequency + trend |
| `trend_only` | 0 | 0 | 3.0 | 1.5 | Pure trend confounding |

### Generate synthetic data

Data is generated on-the-fly during training. To generate and inspect a batch:

```python
from data.dgp import make_dgp

dgp = make_dgp('strong_temporal', n_covariates=5, seed=42)
batch = dgp.sample_batch(n_patients=256, window_hours=48.0)

print(batch.true_ATE)          # ~0.5
print(batch.A.mean())          # ~0.5 (calibrated treatment rate)
print(batch.f_freq[:5])        # observation frequencies
print(batch.trend[:5])         # recency trend scores
```

Run the built-in sanity check:

```bash
python data/dgp.py
```

## Training

Training jointly optimizes the temporal encoder and LoRA adapters using three losses:

- **L_causal** (weight 1.0) — CausalPFN's own loss with known counterfactuals
- **L_reconstruction** (weight 0.5) — BERT-style masked feature prediction
- **L_consistency** (weight 0.1) — smooth embedding updates when new observations arrive

```bash
# Smoke test (CPU, ~2 min)
python scripts/train.py --config strong_temporal --n_steps 5 --batch_size 32 --device cpu

# Full run (GPU, ~1 hour per config)
python scripts/train.py \
    --config strong_temporal \
    --n_steps 3000 \
    --batch_size 64 \
    --checkpoint_dir ./checkpoints/strong_temporal/42 \
    --seed 42 \
    --device cuda
```

### Train all configs and seeds

```bash
for config in tabular_sufficient weak_temporal strong_temporal asymmetric; do
    for seed in 42 123 999; do
        python scripts/train.py \
            --config $config \
            --n_steps 3000 \
            --batch_size 64 \
            --checkpoint_dir ./checkpoints/$config/$seed \
            --seed $seed \
            --device cuda
    done
done
```

Checkpoints are saved to `checkpoint_dir/latest.pt` and include the encoder state, LoRA weights, optimizer state, and scheduler state. Training resumes automatically from existing checkpoints.

## Evaluation

### Main results (encoder vs. static baselines)

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/strong_temporal/42/latest.pt \
    --all_configs \
    --device cuda
```

### Fair comparison (same embeddings to CausalPFN vs. TabPFN-X)

This is the key experiment. It feeds identical features to both estimators to isolate representation quality from estimator quality:

```bash
python scripts/evaluate_fair_comparison.py \
    --checkpoint_dir ./checkpoints \
    --output_dir ./results_fair \
    --device cuda
```

### Trend confounding experiment

Trains and evaluates on the two trend-confounded configs end-to-end:

```bash
python scripts/run_trend_experiment.py --output_dir ./results_trend --device cuda
```

### TabPFN baselines (no encoder)

```bash
python scripts/evaluate_tabpfn.py --output_dir ./results_tabpfn --device cuda
```

## Semi-Synthetic MIMIC-IV Evaluation

This uses real patient trajectories from MIMIC-IV with simulated outcomes from a known structural causal model. Real observation patterns are preserved; only treatment assignment and outcomes are simulated to provide ground-truth counterfactuals.

### Prerequisites

Place the MIMIC-IV demo dataset in `./mimic_iv/` with the standard directory structure:

```
mimic_iv/
  hosp/
    admissions.csv, labevents.csv, d_labitems.csv, patients.csv, ...
  icu/
    chartevents.csv, icustays.csv, inputevents.csv, ...
```

The [MIMIC-IV demo dataset](https://physionet.org/content/mimic-iv-demo/2.2/) is freely available from PhysioNet (no credentialed access required).

### Step 1: Verify cohort extraction

```bash
python data/mimic_cohort.py ./mimic_iv
```

This extracts ~119 ICU stays with 5 time-varying features (heart rate, mean BP, SpO2, creatinine, lactate). Treatment is defined as first IV antibiotic administration. Control patients are assigned pseudo-treatment times sampled from the treated distribution. All covariates are z-scored to match the synthetic DGP convention.

### Step 2: Verify semi-synthetic DGP

```bash
python data/mimic_dgp.py ./mimic_iv
```

This runs all 6 SCM configurations on the extracted cohort and prints ATE accuracy, treatment rates, and temporal feature ranges. The SCM uses the same parametric form as the synthetic DGP so results are directly comparable.

### Step 3: Run evaluation

With trained encoder checkpoints:

```bash
python scripts/evaluate_mimic.py \
    --checkpoint_dir ./checkpoints \
    --mimic_dir ./mimic_iv \
    --output_dir ./results_mimic \
    --device cuda
```

Baselines only (no encoder needed):

```bash
python scripts/evaluate_mimic.py \
    --baselines_only \
    --mimic_dir ./mimic_iv \
    --output_dir ./results_mimic \
    --device cpu
```

The evaluation produces:
- A distribution comparison between MIMIC and synthetic temporal features
- PEHE and ATE relative error for all 4 conditions (CausalPFN/TabPFN-X x Static/Temporal)
- Results saved to JSON and a formatted table

### What the MIMIC evaluation tests

The encoder is trained entirely on synthetic OU-process data. The MIMIC evaluation tests whether its learned representations transfer to real clinical observation patterns. Key differences from synthetic data:

- **Wider observation frequency range** — MIMIC `f_freq` ranges from 1.0 to 7.7 obs/hour vs. synthetic 0.7 to 2.4
- **Fewer observations per patient** — MIMIC median 13 vs. synthetic median 65
- **Partial trend leakage** — SpO2 variability correlates with trend (`r=0.71`) in real data, making the trend confounding experiment harder since static features partially capture the signal
- **Non-Gaussian covariate dynamics** — real vitals do not follow an OU process

## Metrics

- **PEHE** (Precision in Estimation of Heterogeneous Effects): `sqrt(E[(CATE_hat - CATE_true)^2])` — lower is better
- **ATE relative error**: `|ATE_hat - ATE_true| / |ATE_true| * 100%` — lower is better

## Key Results

On `strong_temporal` (heavy frequency confounding):
- Static baselines: 69-97% ATE relative error
- Temporal encoder: 27% ATE relative error (3.6x improvement)

Fair comparison (same embeddings to both estimators):
- TabPFN-X consistently benefits from temporal embeddings (23.1% -> 18.0% ATE error on `strong_temporal`)
- CausalPFN improvement is smaller due to prior distribution mismatch with temporal embedding space

BatchNorm + rank-32 LoRA eliminates the regression where CausalPFN+Temporal performed worse than CausalPFN+Static on the `asymmetric` config.

## Compute

All experiments ran on NYU HPC A100 40GB GPUs. Total: ~12 GPU hours across all experiments. Training a single config/seed takes ~20 minutes on A100.

## Known Issues

- CausalPFN's `clip_outliers()` and `normalize_data()` use in-place tensor operations that corrupt PyTorch's autograd graph. Patched at runtime in `train.py` with `.clone()`-safe equivalents.
- LoRA adapters must be injected before calling `icl_model.to(device)`, or adapter parameters remain on CPU.
- The MIMIC-IV demo dataset contains only 100 patients. Bootstrap resampling is used to generate larger evaluation batches.

## Citation

```bibtex
@article{patel2026temporal,
  title={Temporal Encoder for Irregular Clinical Time-Series: 
         Adapting Causal PFNs via LoRA and Continuous-Time Positional Encodings},
  author={Patel, Maitri and Jian, Darren},
  year={2026}
}
```
