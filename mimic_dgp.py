"""
mimic_dgp.py — Semi-Synthetic DGP Using Real MIMIC-IV Trajectories

Takes real patient histories from MIMIC-IV and overlays simulated treatment
assignments and outcomes from a known structural causal model (SCM). This
gives us ground-truth counterfactuals while preserving real clinical:
    - Irregular observation timestamps
    - Covariate dynamics and marginal distributions
    - Observation frequency patterns
    - Recency trends

The SCM mirrors the synthetic DGP (dgp.py) so results are directly
comparable. The key test: can the temporal encoder, trained on synthetic
OU-process data, recover temporal confounders from real clinical trajectories?

Usage:
    from mimic_cohort import extract_cohort
    from mimic_dgp import MIMICSemiSyntheticDGP, make_mimic_dgp

    cohort = extract_cohort('./mimic_iv', n_covariates=5)
    dgp = make_mimic_dgp('strong_temporal', cohort)
    batch = dgp.generate_batch(n_patients=256)
"""

import numpy as np
from typing import Tuple

from dgp import PatientHistory, DGPBatch
from mimic_cohort import MIMICCohort


class MIMICSemiSyntheticDGP:
    """
    Semi-synthetic DGP that uses real MIMIC-IV histories with simulated outcomes.

    Real components (from MIMIC):
        - Observation timestamps (irregular, clinically-driven)
        - Covariate values (real vital signs and labs, z-scored)
        - Observation frequency (emergent from real monitoring patterns)
        - Recency trend (emergent from real clinical trajectories)

    Simulated components (from SCM):
        - Treatment assignment A (re-simulated to ensure overlap)
        - Potential outcomes Y(0), Y(1) (from parametric SCM)
        - Known ground truth ATE, CATE

    Parameters
    ----------
    cohort : MIMICCohort
        Extracted MIMIC cohort with real PatientHistory objects.
    beta, eta : float
        Coefficients on observation frequency in treatment/outcome models.
    gamma_trend, eta_trend : float
        Coefficients on recency trend in treatment/outcome models.
    theta : float
        True average treatment effect.
    trend_window : float
        Hours before treatment to compute trend (default 6.0).
    sigma_y : float
        Outcome noise scale.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        cohort: MIMICCohort,
        beta: float = 1.0,
        eta: float = 1.0,
        gamma_trend: float = 0.0,
        eta_trend: float = 0.0,
        theta: float = 0.5,
        trend_window: float = 6.0,
        sigma_y: float = 0.3,
        seed: int = 42,
    ):
        self.cohort = cohort
        self.beta = beta
        self.eta = eta
        self.gamma_trend = gamma_trend
        self.eta_trend = eta_trend
        self.theta = theta
        self.trend_window = trend_window
        self.sigma_y = sigma_y
        self.rng = np.random.default_rng(seed)

        n_cov = cohort.histories[0].covariates.shape[1]

        self.alpha = self.rng.normal(0, 0.5, size=n_cov)
        self.rho = self.rng.normal(0, 0.5, size=n_cov)
        self.delta = 0.5
        self.zeta = 0.5

        # Precompute temporal features for the entire cohort
        self._precompute_features()

        # Calibrate treatment intercept
        self.treatment_intercept = self._calibrate_intercept()

    def _precompute_features(self):
        """Compute f_freq, trend, x_bar, and v_i for each patient in the cohort."""
        self._f_freq = np.zeros(len(self.cohort.histories), dtype=np.float32)
        self._trend = np.zeros(len(self.cohort.histories), dtype=np.float32)
        self._x_bar = np.zeros(
            (len(self.cohort.histories), self.cohort.histories[0].covariates.shape[1]),
            dtype=np.float32
        )
        self._v_i = np.zeros(len(self.cohort.histories), dtype=np.float32)

        for i, h in enumerate(self.cohort.histories):
            w = self.cohort.window_hours[i]
            self._f_freq[i] = h.n_obs / max(w, 1.0)
            self._trend[i] = self._compute_trend(h)
            self._x_bar[i] = h.covariates.mean(axis=0)
            # Latent health proxy from covariate severity
            severity = float(np.mean(np.abs(h.covariates.mean(axis=0))))
            self._v_i[i] = 1.0 / (1.0 + np.exp(-severity))

    def _compute_trend(self, history: PatientHistory) -> float:
        """Mean slope of covariates in the last trend_window hours (same as dgp)."""
        recent_mask = history.timestamps >= -self.trend_window
        if recent_mask.sum() < 2:
            n = min(2, history.n_obs)
            t_recent = history.timestamps[-n:]
            x_recent = history.covariates[-n:]
        else:
            t_recent = history.timestamps[recent_mask]
            x_recent = history.covariates[recent_mask]

        t_mean = t_recent.mean()
        t_centered = t_recent - t_mean
        t_var = float((t_centered ** 2).sum())
        if t_var < 1e-8:
            return 0.0

        slopes = []
        for j in range(x_recent.shape[1]):
            x_j = x_recent[:, j]
            slope = float((t_centered * (x_j - x_j.mean())).sum() / t_var)
            slopes.append(slope)
        return float(np.mean(slopes))

    def _calibrate_intercept(self) -> float:
        """Set intercept so treatment rate is ~0.5 across the cohort."""
        raw_logits = []
        for i in range(len(self.cohort.histories)):
            logit = (
                float(self.alpha @ self._x_bar[i])
                + self.beta * self._f_freq[i]
                + self.gamma_trend * self._trend[i]
                + self.delta * self._v_i[i]
            )
            raw_logits.append(logit)
        return -float(np.mean(raw_logits))

    def _assign_treatment(self, i: int) -> int:
        logit = (
            self.treatment_intercept
            + float(self.alpha @ self._x_bar[i])
            + self.beta * self._f_freq[i]
            + self.gamma_trend * self._trend[i]
            + self.delta * self._v_i[i]
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        prob = np.clip(prob, 0.05, 0.95)
        return int(self.rng.binomial(1, prob))

    def _compute_potential_outcomes(self, i: int) -> Tuple[float, float]:
        base = (
            float(self.rho @ self._x_bar[i])
            + self.eta * self._f_freq[i]
            + self.eta_trend * self._trend[i]
            + self.zeta * self._v_i[i]
        )
        noise_0 = self.rng.normal(0, self.sigma_y)
        noise_1 = self.rng.normal(0, self.sigma_y)
        return float(base + noise_0), float(base + self.theta + noise_1)

    def generate_batch(self, n_patients: int = 256) -> DGPBatch:
        """
        Generate a semi-synthetic batch by bootstrap-resampling real MIMIC
        histories and overlaying simulated treatment/outcomes.

        Bootstrap resampling is necessary because the MIMIC demo cohort has
        ~100-130 patients, but evaluation needs larger batches. Each resampled
        patient gets fresh SCM noise.
        """
        n_cohort = len(self.cohort.histories)
        indices = self.rng.choice(n_cohort, size=n_patients, replace=True)

        histories = []
        A_list, Y0_list, Y1_list = [], [], []
        f_freq_list, trend_list = [], []

        for idx in indices:
            histories.append(self.cohort.histories[idx])

            A_i = self._assign_treatment(idx)
            y0_i, y1_i = self._compute_potential_outcomes(idx)

            A_list.append(A_i)
            Y0_list.append(y0_i)
            Y1_list.append(y1_i)
            f_freq_list.append(self._f_freq[idx])
            trend_list.append(self._trend[idx])

        A = np.array(A_list, dtype=np.float32)
        Y0 = np.array(Y0_list, dtype=np.float32)
        Y1 = np.array(Y1_list, dtype=np.float32)
        Y_obs = np.where(A == 1, Y1, Y0)
        f_freq = np.array(f_freq_list, dtype=np.float32)
        trend = np.array(trend_list, dtype=np.float32)

        return DGPBatch(
            histories=histories,
            A=A, Y0=Y0, Y1=Y1, Y_obs=Y_obs,
            true_ATE=float((Y1 - Y0).mean()),
            true_CATE=Y1 - Y0,
            f_freq=f_freq,
            trend=trend,
        )


# ------------------------------------------------------------------
# Configuration registry — mirrors dgp.py
# ------------------------------------------------------------------

MIMIC_DGP_CONFIGS = {
    "tabular_sufficient":    dict(beta=0.0, eta=0.0,
                                  gamma_trend=0.0, eta_trend=0.0),
    "weak_temporal":         dict(beta=0.5, eta=0.5,
                                  gamma_trend=0.0, eta_trend=0.0),
    "strong_temporal":       dict(beta=2.0, eta=2.0,
                                  gamma_trend=0.0, eta_trend=0.0),
    "asymmetric":            dict(beta=2.0, eta=0.0,
                                  gamma_trend=0.0, eta_trend=0.0),
    "strong_temporal_trend": dict(beta=2.0, eta=2.0,
                                  gamma_trend=2.0, eta_trend=2.0),
    "trend_only":            dict(beta=0.0, eta=0.0,
                                  gamma_trend=3.0, eta_trend=1.5),
}


def make_mimic_dgp(
    config_name: str,
    cohort: MIMICCohort,
    seed: int = 42,
) -> MIMICSemiSyntheticDGP:
    """Factory function for semi-synthetic MIMIC DGP configurations."""
    if config_name not in MIMIC_DGP_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Choose from: {list(MIMIC_DGP_CONFIGS.keys())}"
        )
    return MIMICSemiSyntheticDGP(
        cohort=cohort,
        seed=seed,
        **MIMIC_DGP_CONFIGS[config_name],
    )


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from mimic_cohort import extract_cohort

    mimic_dir = sys.argv[1] if len(sys.argv) > 1 else './mimic_iv'
    cohort = extract_cohort(mimic_dir, n_covariates=5)

    print(f"\n{'='*60}")
    print("Semi-synthetic DGP sanity check")
    print(f"{'='*60}")

    for config_name in MIMIC_DGP_CONFIGS:
        dgp = make_mimic_dgp(config_name, cohort, seed=42)
        batch = dgp.generate_batch(n_patients=300)

        obs_counts = [h.n_obs for h in batch.histories]
        print(f"\nConfig: {config_name}")
        print(f"  True ATE:       {batch.true_ATE:.4f}  (target ~{dgp.theta:.2f})")
        print(f"  Treatment rate: {batch.A.mean():.3f}")
        print(f"  Obs/patient:    mean={np.mean(obs_counts):.1f}")
        print(f"  f_freq range:   [{batch.f_freq.min():.2f}, {batch.f_freq.max():.2f}]")
        print(f"  Trend range:    [{batch.trend.min():.3f}, {batch.trend.max():.3f}]")

    print(f"\nSanity check complete.")
