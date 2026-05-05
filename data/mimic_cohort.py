"""
mimic_cohort.py — MIMIC-IV Cohort Extraction for Semi-Synthetic Evaluation

Extracts irregular clinical time-series from MIMIC-IV demo data:
    - Treatment: first IV antibiotic administration
    - Features: vitals (HR, MAP, SpO2, RR, Temp) + labs (lactate, creatinine)
    - Window: ICU admission to first antibiotic (or pseudo-treatment time for controls)

Outputs PatientHistory objects compatible with the existing temporal encoder.

Usage:
    from mimic_cohort import extract_cohort
    cohort = extract_cohort('./mimic_iv', n_covariates=5)
    print(f'{len(cohort.histories)} patients extracted')
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from dgp import PatientHistory


# ------------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------------

VITAL_ITEMS = {
    220045: 'heart_rate',
    220181: 'mean_bp',
    220277: 'spo2',
    220210: 'resp_rate',
    223761: 'temperature',
}

LAB_ITEMS = {
    50813: 'lactate',
    50912: 'creatinine',
    50971: 'potassium',
    50983: 'sodium',
    50931: 'glucose',
    51222: 'hemoglobin',
    51265: 'platelets',
    51301: 'wbc',
    51006: 'bun',
    50882: 'bicarbonate',
    50902: 'chloride',
    50885: 'bilirubin_total',
}

FEATURE_SETS = {
    5:  ['heart_rate', 'mean_bp', 'spo2', 'creatinine', 'lactate'],
    10: ['heart_rate', 'mean_bp', 'spo2', 'resp_rate', 'temperature',
         'lactate', 'creatinine', 'potassium', 'sodium', 'glucose'],
    17: ['heart_rate', 'mean_bp', 'spo2', 'resp_rate', 'temperature',
         'lactate', 'creatinine', 'potassium', 'sodium', 'glucose',
         'hemoglobin', 'platelets', 'wbc', 'bun', 'bicarbonate',
         'chloride', 'bilirubin_total'],
}

# Clinically reasonable ranges for outlier clipping
VALID_RANGES = {
    'heart_rate':      (20, 250),
    'mean_bp':         (20, 200),
    'spo2':            (50, 100),
    'resp_rate':       (4, 60),
    'temperature':     (90, 110),  # Fahrenheit
    'lactate':         (0.1, 30),
    'creatinine':      (0.1, 25),
    'potassium':       (1.5, 10),
    'sodium':          (110, 170),
    'glucose':         (20, 600),
    'hemoglobin':      (3, 20),
    'platelets':       (5, 1000),
    'wbc':             (0.1, 100),
    'bun':             (1, 200),
    'bicarbonate':     (5, 50),
    'chloride':        (70, 140),
    'bilirubin_total': (0.1, 50),
}


@dataclass
class MIMICCohort:
    """Extracted cohort ready for semi-synthetic outcome simulation."""
    histories: List[PatientHistory]
    stay_ids: List[int]
    treated: np.ndarray
    window_hours: np.ndarray
    feature_names: List[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    n_excluded: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _load_icu_stays(mimic_dir: str) -> pd.DataFrame:
    df = pd.read_csv(Path(mimic_dir) / 'icu' / 'icustays.csv')
    df['intime'] = pd.to_datetime(df['intime'])
    df['outtime'] = pd.to_datetime(df['outtime'])
    return df


def _load_first_antibiotic_times(mimic_dir: str) -> Dict[int, pd.Timestamp]:
    ie = pd.read_csv(Path(mimic_dir) / 'icu' / 'inputevents.csv',
                     usecols=['stay_id', 'starttime', 'ordercategoryname'])
    abx = ie[ie['ordercategoryname'].str.contains('Antibiotic', case=False, na=False)]
    abx = abx.copy()
    abx['starttime'] = pd.to_datetime(abx['starttime'])
    first = abx.groupby('stay_id')['starttime'].min()
    return first.to_dict()


def _load_vitals(mimic_dir: str, stay_ids: set, item_map: Dict[int, str]
                 ) -> Dict[int, List[Tuple[pd.Timestamp, str, float]]]:
    ce = pd.read_csv(Path(mimic_dir) / 'icu' / 'chartevents.csv',
                     usecols=['stay_id', 'charttime', 'itemid', 'valuenum'])
    ce = ce[ce['itemid'].isin(item_map) & ce['valuenum'].notna()
            & ce['stay_id'].isin(stay_ids)]
    ce['charttime'] = pd.to_datetime(ce['charttime'])
    ce['feature'] = ce['itemid'].map(item_map)

    records: Dict[int, list] = {sid: [] for sid in stay_ids}
    for _, row in ce.iterrows():
        records[row['stay_id']].append(
            (row['charttime'], row['feature'], float(row['valuenum']))
        )
    return records


def _load_labs(mimic_dir: str, hadm_to_stay: Dict[int, int],
               item_map: Dict[int, str]
               ) -> Dict[int, List[Tuple[pd.Timestamp, str, float]]]:
    le = pd.read_csv(Path(mimic_dir) / 'hosp' / 'labevents.csv',
                     usecols=['hadm_id', 'charttime', 'itemid', 'valuenum'])
    valid_hadms = set(hadm_to_stay.keys())
    le = le[le['itemid'].isin(item_map) & le['valuenum'].notna()
            & le['hadm_id'].isin(valid_hadms)]
    le['charttime'] = pd.to_datetime(le['charttime'])
    le['feature'] = le['itemid'].map(item_map)

    records: Dict[int, list] = {}
    for _, row in le.iterrows():
        sid = hadm_to_stay[row['hadm_id']]
        if sid not in records:
            records[sid] = []
        records[sid].append(
            (row['charttime'], row['feature'], float(row['valuenum']))
        )
    return records


# ------------------------------------------------------------------
# Observation matrix construction
# ------------------------------------------------------------------

def _build_observation_matrix(
    vital_records: List[Tuple[pd.Timestamp, str, float]],
    lab_records: List[Tuple[pd.Timestamp, str, float]],
    feature_names: List[str],
    intime: pd.Timestamp,
    cutoff_time: pd.Timestamp,
    valid_ranges: Dict[str, Tuple[float, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Merge vitals and labs into a single observation matrix.
    Uses last-observation-carried-forward (LOCF) for missing features.

    Returns (timestamps_hours, covariates) or None if insufficient data.
    """
    all_records = vital_records + lab_records

    # Filter to pre-treatment window
    window_records = [
        (t, feat, val) for t, feat, val in all_records
        if intime <= t < cutoff_time
    ]
    if len(window_records) < 3:
        return None

    # Clip outliers per feature
    clipped = []
    for t, feat, val in window_records:
        if feat in valid_ranges:
            lo, hi = valid_ranges[feat]
            val = max(lo, min(hi, val))
        clipped.append((t, feat, val))

    # Sort by time
    clipped.sort(key=lambda x: x[0])

    # Group by unique timestamps (within 1 minute)
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    d = len(feature_names)

    grouped_times = []
    grouped_values = []
    current_group_time = None
    current_values = {}

    for t, feat, val in clipped:
        if feat not in feat_idx:
            continue
        if current_group_time is None or (t - current_group_time).total_seconds() > 60:
            if current_values:
                grouped_times.append(current_group_time)
                grouped_values.append(dict(current_values))
            current_group_time = t
            current_values = {}
        current_values[feat] = val

    if current_values:
        grouped_times.append(current_group_time)
        grouped_values.append(dict(current_values))

    if len(grouped_times) < 3:
        return None

    # Build matrix with LOCF
    n_times = len(grouped_times)
    covariates = np.full((n_times, d), np.nan, dtype=np.float32)
    last_known = np.full(d, np.nan, dtype=np.float32)

    for i, vals in enumerate(grouped_values):
        for feat, val in vals.items():
            j = feat_idx[feat]
            last_known[j] = val
        covariates[i] = last_known.copy()

    # Drop rows where no feature has ever been observed
    any_valid = np.any(~np.isnan(covariates), axis=1)
    valid_idx = np.where(any_valid)[0]
    if len(valid_idx) < 3:
        return None

    covariates = covariates[valid_idx]
    grouped_times = [grouped_times[i] for i in valid_idx]

    # Forward-fill remaining NaNs from column medians as fallback
    for j in range(d):
        col = covariates[:, j]
        if np.all(np.isnan(col)):
            covariates[:, j] = 0.0
            continue
        valid_vals = col[~np.isnan(col)]
        fill_val = valid_vals[0]
        for i in range(len(col)):
            if np.isnan(col[i]):
                col[i] = fill_val
            else:
                fill_val = col[i]

    # Convert timestamps to hours relative to cutoff (negative values)
    window_hours = (cutoff_time - intime).total_seconds() / 3600
    timestamps = np.array([
        (t - cutoff_time).total_seconds() / 3600 for t in grouped_times
    ], dtype=np.float32)

    return timestamps, covariates


# ------------------------------------------------------------------
# Main extraction
# ------------------------------------------------------------------

def extract_cohort(
    mimic_dir: str,
    n_covariates: int = 5,
    min_obs: int = 5,
    min_window_hours: float = 1.0,
    seed: int = 42,
) -> MIMICCohort:
    """
    Extract a semi-synthetic cohort from MIMIC-IV demo data.

    Parameters
    ----------
    mimic_dir : str
        Path to the MIMIC-IV directory containing hosp/ and icu/ subdirs.
    n_covariates : int
        Number of features to extract (5, 10, or 17).
    min_obs : int
        Minimum number of observations required per patient.
    min_window_hours : float
        Minimum pre-treatment window length in hours.
    seed : int
        Random seed for pseudo-treatment time assignment.

    Returns
    -------
    MIMICCohort with real PatientHistory objects.
    """
    rng = np.random.default_rng(seed)

    if n_covariates not in FEATURE_SETS:
        raise ValueError(f"n_covariates must be one of {list(FEATURE_SETS.keys())}")
    feature_names = FEATURE_SETS[n_covariates]

    # Determine which items to load
    needed_vitals = {iid: name for iid, name in VITAL_ITEMS.items()
                     if name in feature_names}
    needed_labs = {iid: name for iid, name in LAB_ITEMS.items()
                   if name in feature_names}

    print(f"Extracting MIMIC cohort: {n_covariates} features")
    print(f"  Vitals: {list(needed_vitals.values())}")
    print(f"  Labs: {list(needed_labs.values())}")

    # Load tables
    icu_stays = _load_icu_stays(mimic_dir)
    abx_times = _load_first_antibiotic_times(mimic_dir)

    stay_ids = set(icu_stays['stay_id'])
    hadm_to_stay = dict(zip(icu_stays['hadm_id'], icu_stays['stay_id']))

    vital_records = _load_vitals(mimic_dir, stay_ids, needed_vitals) if needed_vitals else {}
    lab_records = _load_labs(mimic_dir, hadm_to_stay, needed_labs) if needed_labs else {}

    # Assign treatment times
    treated_windows = []
    for _, row in icu_stays.iterrows():
        sid = row['stay_id']
        if sid in abx_times:
            window_h = (abx_times[sid] - row['intime']).total_seconds() / 3600
            if window_h > min_window_hours:
                treated_windows.append(window_h)

    treated_windows_arr = np.array(treated_windows) if treated_windows else np.array([6.0])

    # Process each stay
    histories = []
    stay_id_list = []
    treated_list = []
    window_hours_list = []
    exclusion_reasons = {
        'abx_before_icu': 0,
        'window_too_short': 0,
        'insufficient_obs': 0,
    }

    for _, row in icu_stays.iterrows():
        sid = row['stay_id']
        intime = row['intime']
        los_hours = row['los'] * 24

        # Determine cutoff time
        is_treated = sid in abx_times
        if is_treated:
            cutoff = abx_times[sid]
            window_h = (cutoff - intime).total_seconds() / 3600
            if window_h < min_window_hours:
                if window_h < 0:
                    exclusion_reasons['abx_before_icu'] += 1
                else:
                    exclusion_reasons['window_too_short'] += 1
                continue
        else:
            pseudo_h = rng.choice(treated_windows_arr)
            pseudo_h = min(pseudo_h, los_hours - 0.5)
            pseudo_h = max(pseudo_h, min_window_hours)
            cutoff = intime + pd.Timedelta(hours=pseudo_h)

        # Build observation matrix
        v_recs = vital_records.get(sid, [])
        l_recs = lab_records.get(sid, [])
        result = _build_observation_matrix(
            v_recs, l_recs, feature_names, intime, cutoff, VALID_RANGES
        )
        if result is None or len(result[0]) < min_obs:
            exclusion_reasons['insufficient_obs'] += 1
            continue

        timestamps, covariates = result
        histories.append(PatientHistory(
            timestamps=timestamps,
            covariates=covariates,
            n_obs=len(timestamps),
        ))
        stay_id_list.append(sid)
        treated_list.append(1 if is_treated else 0)
        window_hours_list.append(
            (cutoff - intime).total_seconds() / 3600
        )

    n_excluded = sum(exclusion_reasons.values())
    print(f"\n  Included: {len(histories)} stays")
    print(f"  Excluded: {n_excluded} stays")
    for reason, count in exclusion_reasons.items():
        if count > 0:
            print(f"    {reason}: {count}")

    # Compute population-level statistics for z-scoring
    all_values = np.concatenate([h.covariates for h in histories], axis=0)
    feat_means = all_values.mean(axis=0)
    feat_stds = all_values.std(axis=0) + 1e-6

    # Z-score all covariates
    for h in histories:
        h.covariates = ((h.covariates - feat_means) / feat_stds).astype(np.float32)

    obs_counts = [h.n_obs for h in histories]
    print(f"\n  Observations per patient: "
          f"min={min(obs_counts)}, median={np.median(obs_counts):.0f}, "
          f"max={max(obs_counts)}")
    print(f"  Treatment rate: {np.mean(treated_list):.3f}")
    print(f"  Window hours: median={np.median(window_hours_list):.1f}, "
          f"range=[{min(window_hours_list):.1f}, {max(window_hours_list):.1f}]")

    return MIMICCohort(
        histories=histories,
        stay_ids=stay_id_list,
        treated=np.array(treated_list, dtype=np.float32),
        window_hours=np.array(window_hours_list, dtype=np.float32),
        feature_names=feature_names,
        feature_means=feat_means,
        feature_stds=feat_stds,
        n_excluded=n_excluded,
        exclusion_reasons=exclusion_reasons,
    )


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    mimic_dir = sys.argv[1] if len(sys.argv) > 1 else './mimic_iv'
    cohort = extract_cohort(mimic_dir, n_covariates=5)

    print(f"\n{'='*50}")
    print("Sample patient:")
    h = cohort.histories[0]
    print(f"  timestamps: {h.timestamps[:5]}... (n={h.n_obs})")
    print(f"  covariates shape: {h.covariates.shape}")
    print(f"  covariate range: [{h.covariates.min():.2f}, {h.covariates.max():.2f}]")
