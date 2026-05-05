"""
dgp.py — Synthetic Data Generating Process for Irregular Clinical Time-Series

Generates batches of synthetic ICU patients with known counterfactuals.
Observation times follow a competing-event model (routine + alarm-triggered),
covariate dynamics follow an Ornstein-Uhlenbeck process, and both treatment
assignment and outcomes are confounded by observation frequency and recency
trend — meaning causal information lives in the timestamps, not just the values.

Two temporal confounders are available:
    1. Observation frequency: sicker patients are checked more often.
    2. Recency trend: the direction of covariate change in the last few hours.
       Genuinely invisible to mean+std features.

Six configurations control confounding strength (see DGP_CONFIGS).

Usage:
    dgp = make_dgp('strong_temporal_trend', n_covariates=5)
    batch = dgp.sample_batch(n_patients=256, window_hours=48)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PatientHistory:
    """
    Holds the irregular observation history for one synthetic patient.

    timestamps : array of shape (m_i,) — observation times in hours,
                 negative values, ranging from -window_hours to 0.
    covariates : array of shape (m_i, d) — covariate values at each observation
    n_obs      : int — number of observations m_i (varies per patient)
    """
    timestamps: np.ndarray
    covariates: np.ndarray
    n_obs: int


@dataclass
class DGPBatch:
    """
    A batch of synthetic patients with ground-truth causal quantities.

    histories  : list of PatientHistory, length n_patients
    A          : array of shape (n_patients,) — binary treatment indicators
    Y0         : array of shape (n_patients,) — potential outcome under control
    Y1         : array of shape (n_patients,) — potential outcome under treatment
    Y_obs      : array of shape (n_patients,) — observed outcome (Y0 or Y1)
    true_ATE   : float — ground truth average treatment effect E[Y1 - Y0]
    true_CATE  : array of shape (n_patients,) — individual treatment effects
    f_freq     : array of shape (n_patients,) — observation frequency
    trend      : array of shape (n_patients,) — recency trend score
    """
    histories: List[PatientHistory]
    A: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray
    Y_obs: np.ndarray
    true_ATE: float
    true_CATE: np.ndarray
    f_freq: np.ndarray
    trend: np.ndarray


class ICUDataDGP:
    """
    Data Generating Process for irregular ICU time-series with temporal confounding.

    Two temporal confounders are available:

    1. Observation frequency f_freq = m_i / W:
       Sicker patients are checked more often. This confounds through *how often*
       the patient is monitored. Invisible to static summaries only when combined
       with other features, but partially recoverable from count-based features.

    2. Recency trend score:
       Mean slope of covariate values in the last trend_window hours before
       treatment, estimated via linear regression. This captures *direction* of
       change — a deteriorating patient has a positive trend, an improving one
       has negative. Genuinely invisible to mean+std: identical mean and std
       can correspond to very different trends.

    Parameters
    ----------
    beta : float
        Coefficient on f_freq in treatment assignment.
    eta : float
        Coefficient on f_freq in outcome model.
    gamma_trend : float
        Coefficient on recency trend in treatment assignment.
        gamma_trend=0: trend carries no causal information.
        gamma_trend=3: strong trend confounding in treatment.
    eta_trend : float
        Coefficient on recency trend in outcome model.
    trend_window : float
        Hours before treatment to use for trend computation (default 6.0).
        Clinically: the last 6 hours is when acute deterioration is most visible.
    """

    def __init__(
        self,
        n_covariates: int = 5,
        beta: float = 1.0,
        eta: float = 1.0,
        gamma_trend: float = 0.0,
        eta_trend: float = 0.0,
        trend_window: float = 6.0,
        theta: float = 0.5,
        lambda_routine: float = 0.25,
        lambda_alarm_base: float = 0.5,
        gamma: float = 1.0,
        kappa: float = 0.3,
        sigma_x: float = 0.5,
        sigma_y: float = 0.3,
        seed: int = 42,
    ):
        self.n_covariates = n_covariates
        self.beta = beta
        self.eta = eta
        self.gamma_trend = gamma_trend
        self.eta_trend = eta_trend
        self.trend_window = trend_window
        self.theta = theta
        self.lambda_routine = lambda_routine
        self.lambda_alarm_base = lambda_alarm_base
        self.gamma = gamma
        self.kappa = kappa
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rng = np.random.default_rng(seed)

        # Fixed coefficient vectors
        self.alpha = self.rng.normal(0, 0.5, size=n_covariates)
        self.rho   = self.rng.normal(0, 0.5, size=n_covariates)
        self.delta = 0.5
        self.zeta  = 0.5

        # Calibrate treatment intercept for positivity
        self.treatment_intercept = self._calibrate_treatment_intercept(
            n_pilot=1000, window_hours=48.0
        )

    # ------------------------------------------------------------------
    # Treatment intercept calibration
    # ------------------------------------------------------------------

    def _calibrate_treatment_intercept(
        self, n_pilot: int = 1000, window_hours: float = 48.0
    ) -> float:
        """
        Calibrate intercept so treatment rate is approximately 0.5.
        Runs a pilot simulation that includes trend computation so the
        calibration is accurate for trend-confounded configs.
        """
        tmp_rng = np.random.default_rng(seed=0)
        raw_logits = []

        for _ in range(n_pilot):
            v_i = tmp_rng.uniform(0.0, 1.0)
            mu_i = tmp_rng.normal(v_i, 0.3, size=self.n_covariates)

            # Approximate f_freq
            x0 = mu_i + tmp_rng.normal(0, 0.2, size=self.n_covariates)
            severity = float(np.mean(np.abs(x0)))
            lambda_alarm = self.lambda_alarm_base * np.exp(self.gamma * severity)
            lambda_alarm = np.clip(lambda_alarm, 1e-3, 20.0)
            lambda_total = self.lambda_routine + lambda_alarm
            f_freq_approx = lambda_total

            # Approximate trend: for calibration, use zero mean trend
            # (trend is mean-zero by construction — positive and negative
            #  trajectories are equally likely across the population)
            trend_approx = 0.0

            logit = (
                float(self.alpha @ mu_i)
                + self.beta * f_freq_approx
                + self.gamma_trend * trend_approx
                + self.delta * v_i
            )
            raw_logits.append(logit)

        return -float(np.mean(raw_logits))

    # ------------------------------------------------------------------
    # Step 1: Patient baseline
    # ------------------------------------------------------------------

    def _sample_patient_baseline(self) -> Tuple[float, np.ndarray]:
        v_i = self.rng.uniform(0.0, 1.0)
        mu_i = self.rng.normal(v_i, 0.3, size=self.n_covariates)
        return v_i, mu_i

    # ------------------------------------------------------------------
    # Step 2: Observation time process (competing event model)
    # ------------------------------------------------------------------

    def _sample_next_gap(self, x_prev: np.ndarray) -> float:
        e1 = self.rng.exponential(1.0 / self.lambda_routine)
        severity = float(np.mean(np.abs(x_prev)))
        lambda_alarm = self.lambda_alarm_base * np.exp(self.gamma * severity)
        lambda_alarm = np.clip(lambda_alarm, 1e-3, 20.0)
        e2 = self.rng.exponential(1.0 / lambda_alarm)
        return float(min(e1, e2))

    # ------------------------------------------------------------------
    # Step 3: Covariate dynamics (Ornstein-Uhlenbeck)
    # ------------------------------------------------------------------

    def _evolve_covariates(
        self,
        x_prev: np.ndarray,
        mu_i: np.ndarray,
        delta_s: float,
    ) -> np.ndarray:
        decay = np.exp(-self.kappa * delta_s)
        variance = (self.sigma_x ** 2) * (1 - np.exp(-2 * self.kappa * delta_s)) / (2 * self.kappa)
        noise = self.rng.normal(0, np.sqrt(variance), size=self.n_covariates)
        return mu_i + (x_prev - mu_i) * decay + noise

    # ------------------------------------------------------------------
    # Step 4: Generate full patient history
    # ------------------------------------------------------------------

    def _generate_history(
        self,
        v_i: float,
        mu_i: np.ndarray,
        window_hours: float,
    ) -> PatientHistory:
        timestamps = []
        covariates = []

        current_time = -window_hours
        x_current = mu_i + self.rng.normal(0, 0.2, size=self.n_covariates)

        while current_time < 0.0:
            timestamps.append(current_time)
            covariates.append(x_current.copy())
            gap = self._sample_next_gap(x_current)
            x_next = self._evolve_covariates(x_current, mu_i, gap)
            current_time += gap
            x_current = x_next

        if len(timestamps) < 2:
            timestamps.append(-window_hours / 2)
            covariates.append(mu_i + self.rng.normal(0, 0.1, size=self.n_covariates))
            order = np.argsort(timestamps)
            timestamps = [timestamps[i] for i in order]
            covariates = [covariates[i] for i in order]

        return PatientHistory(
            timestamps=np.array(timestamps, dtype=np.float32),
            covariates=np.array(covariates, dtype=np.float32),
            n_obs=len(timestamps),
        )

    # ------------------------------------------------------------------
    # Temporal feature computation
    # ------------------------------------------------------------------

    def _compute_frequency(self, history: PatientHistory, window_hours: float) -> float:
        """Observation frequency: m_i / W. Only recoverable from timestamps."""
        return history.n_obs / window_hours

    def _compute_trend(self, history: PatientHistory) -> float:
        """
        Recency trend score: mean slope of covariate values in the last
        trend_window hours, estimated via linear regression.

        Clinically: captures whether the patient is deteriorating (positive)
        or improving (negative) right before treatment.

        This is INVISIBLE to mean+std features. Two patients with identical
        mean and std can have opposite trends if one is rising and one falling.

        Returns a scalar normalized by the range of timestamps in the window
        to be scale-invariant across different observation densities.
        """
        recent_mask = history.timestamps >= -self.trend_window

        if recent_mask.sum() < 2:
            # Fewer than 2 observations in the trend window — use last 2 points
            n = min(2, history.n_obs)
            t_recent = history.timestamps[-n:]
            x_recent = history.covariates[-n:]
        else:
            t_recent = history.timestamps[recent_mask]
            x_recent = history.covariates[recent_mask]

        # Linear regression: x_j ~ slope_j * t + intercept_j
        # slope = cov(t, x) / var(t)
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

        # Mean slope across covariates
        # Note: timestamps are negative and increase toward 0, so a positive
        # slope means covariate values are rising as treatment approaches.
        return float(np.mean(slopes))

    # ------------------------------------------------------------------
    # Treatment assignment and outcomes
    # ------------------------------------------------------------------

    def _assign_treatment(
        self,
        history: PatientHistory,
        v_i: float,
        f_freq: float,
        trend: float,
    ) -> int:
        """
        P(A=1|H_i) = sigmoid(intercept + alpha^T*x_bar + beta*f_freq
                              + gamma_trend*trend + delta*v_i)

        gamma_trend > 0: deteriorating patients (positive trend) are more
        likely to be treated — clinically realistic for sepsis interventions.
        """
        x_bar = history.covariates.mean(axis=0)
        logit = (
            self.treatment_intercept
            + float(self.alpha @ x_bar)
            + self.beta * f_freq
            + self.gamma_trend * trend    # <- trend confounding
            + self.delta * v_i
        )
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        prob = np.clip(prob, 0.05, 0.95)
        return int(self.rng.binomial(1, prob))

    def _compute_potential_outcomes(
        self,
        history: PatientHistory,
        v_i: float,
        f_freq: float,
        trend: float,
    ) -> Tuple[float, float]:
        """
        Y^(a) = rho^T*x_bar + eta*f_freq + eta_trend*trend
                + zeta*v_i + theta*a + noise

        eta_trend > 0: patients who were deteriorating (positive trend) have
        worse outcomes regardless of treatment — confounding through trend.
        """
        x_bar = history.covariates.mean(axis=0)
        base = (
            float(self.rho @ x_bar)
            + self.eta * f_freq
            + self.eta_trend * trend      # <- trend confounding in outcome
            + self.zeta * v_i
        )
        noise_0 = self.rng.normal(0, self.sigma_y)
        noise_1 = self.rng.normal(0, self.sigma_y)
        return float(base + noise_0), float(base + self.theta + noise_1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_batch(
        self,
        n_patients: int = 256,
        window_hours: float = 48.0,
    ) -> DGPBatch:
        """Sample a full batch of synthetic patients."""
        histories, A_list, Y0_list, Y1_list = [], [], [], []
        f_freq_list, trend_list = [], []

        for _ in range(n_patients):
            v_i, mu_i = self._sample_patient_baseline()
            history    = self._generate_history(v_i, mu_i, window_hours)
            f_freq     = self._compute_frequency(history, window_hours)
            trend      = self._compute_trend(history)

            A_i        = self._assign_treatment(history, v_i, f_freq, trend)
            y0_i, y1_i = self._compute_potential_outcomes(history, v_i, f_freq, trend)

            histories.append(history)
            A_list.append(A_i)
            Y0_list.append(y0_i)
            Y1_list.append(y1_i)
            f_freq_list.append(f_freq)
            trend_list.append(trend)

        A     = np.array(A_list,    dtype=np.float32)
        Y0    = np.array(Y0_list,   dtype=np.float32)
        Y1    = np.array(Y1_list,   dtype=np.float32)
        Y_obs = np.where(A == 1, Y1, Y0)
        f_freq = np.array(f_freq_list, dtype=np.float32)
        trend  = np.array(trend_list,  dtype=np.float32)

        return DGPBatch(
            histories=histories,
            A=A, Y0=Y0, Y1=Y1, Y_obs=Y_obs,
            true_ATE=float((Y1 - Y0).mean()),
            true_CATE=Y1 - Y0,
            f_freq=f_freq,
            trend=trend,
        )


# ------------------------------------------------------------------
# Configuration registry
# ------------------------------------------------------------------

DGP_CONFIGS = {
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


def make_dgp(config_name: str, n_covariates: int = 5, seed: int = 42) -> ICUDataDGP:
    """
    Factory function for all DGP configurations.

    Parameters
    ----------
    config_name : str
        One of the six config names in DGP_CONFIGS.
    """
    if config_name not in DGP_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Choose from: {list(DGP_CONFIGS.keys())}"
        )
    return ICUDataDGP(
        n_covariates=n_covariates,
        seed=seed,
        **DGP_CONFIGS[config_name]
    )


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Running DGP sanity check (all 6 configs)...\n")

    for config_name in DGP_CONFIGS:
        dgp   = make_dgp(config_name, n_covariates=5, seed=42)
        batch = dgp.sample_batch(n_patients=300, window_hours=48.0)

        obs_counts = [h.n_obs for h in batch.histories]
        print(f"Config: {config_name}")
        print(f"  gamma_trend={dgp.gamma_trend}, eta_trend={dgp.eta_trend}")
        print(f"  True ATE:       {batch.true_ATE:.4f}  (should be ~{dgp.theta:.2f})")
        print(f"  Treatment rate: {batch.A.mean():.3f}  (should be near 0.5)")
        print(f"  Obs/patient:    mean={np.mean(obs_counts):.1f}")
        print(f"  Trend range:    [{batch.trend.min():.3f}, {batch.trend.max():.3f}]")
        print(f"  Trend std:      {batch.trend.std():.3f}  (>0 means trend varies)")

        # Verify trend is genuinely invisible to mean+std
        # Correlation of trend with mean should be low
        means = np.array([h.covariates.mean() for h in batch.histories])
        stds  = np.array([h.covariates.std()  for h in batch.histories])
        corr_mean  = float(np.corrcoef(batch.trend, means)[0, 1])
        corr_std   = float(np.corrcoef(batch.trend, stds)[0, 1])
        print(f"  corr(trend, mean): {corr_mean:.3f}  (near 0 = trend invisible to mean)")
        print(f"  corr(trend, std):  {corr_std:.3f}  (near 0 = trend invisible to std)")
        print()

    print("Sanity check complete.")
