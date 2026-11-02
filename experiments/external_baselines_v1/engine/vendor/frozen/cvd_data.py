"""Read and encode CVD covariates for a fixed empirical-population benchmark.

Only baseline candidate columns listed below are retained.  Their measurement
timing still requires confirmation from the source data dictionary.  Neither
observed outcomes nor fitted risk scores enter the basis.  Preprocessing uses
the entire fixed covariate population, so this is a transductive method benchmark
and is not an evaluation of generalization to new patients.
"""
from __future__ import annotations

import csv
import gzip
import io
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


REQUIRED_NUMERIC = ("age", "sbp", "bmi", "tc", "hdlc", "eGFR")
OPTIONAL_NUMERIC = ("ADI_NATRANK",)
CATEGORICAL = ("sex", "race", "diabetes", "smoker", "insurance_type")
OBSERVED_ACTIONS = ("statin", "antihtn")
MISSING_TOKENS = frozenset(("", "na", "n/a", "nan", "null", "<na>"))
BASELINE_COLUMNS = REQUIRED_NUMERIC + OPTIONAL_NUMERIC + CATEGORICAL
READ_COLUMNS = ("PATID", "INDEX_DATE") + BASELINE_COLUMNS + OBSERVED_ACTIONS


def _missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


def _resolve_source(input_path, member):
    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError("Input does not exist: %s" % path)
    if path.is_dir():
        if member is not None:
            candidate = path / member
            if not candidate.is_file():
                raise FileNotFoundError("Requested CSV member is absent from the input directory.")
            path = candidate
        else:
            candidates = [p for p in (path / "strict_cohort2/all_data.csv", path / "all_data.csv") if p.is_file()]
            if len(candidates) != 1:
                raise ValueError("Directory must contain exactly one default all_data.csv; use --member to select a file.")
            path = candidates[0]
        return path, None
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [i.filename for i in archive.infolist() if not i.is_dir()]
            if member is None:
                preferred = "strict_cohort2/all_data.csv"
                if preferred in names:
                    chosen = preferred
                else:
                    candidates = [name for name in names if Path(name).name == "all_data.csv"]
                    if len(candidates) != 1:
                        raise ValueError("ZIP must contain one identifiable all_data.csv; use --member for an explicit member.")
                    chosen = candidates[0]
            else:
                chosen = member
            if names.count(chosen) != 1:
                raise ValueError("Requested ZIP member is absent or duplicated.")
        return path, chosen
    if member is not None:
        raise ValueError("--member is only applicable to a ZIP archive or an extracted directory.")
    return path, None


@contextmanager
def _text_source(path, zip_member):
    if zip_member is not None:
        with zipfile.ZipFile(path) as archive:
            with archive.open(zip_member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                    yield stream
    elif str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            yield stream
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield stream


def _read_selected(path, zip_member):
    """Scan every CSV record, rejecting malformed widths without printing cells."""
    rows = []
    bad_width = 0
    with _text_source(path, zip_member) as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        while header == []:
            header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty.")
        header = [name.strip() for name in header]
        duplicates = [name for name, count in Counter(header).items() if count > 1]
        if duplicates:
            raise ValueError("CSV contains %d duplicate column names after whitespace trimming." % len(duplicates))
        required = ("PATID",) + REQUIRED_NUMERIC
        absent = [name for name in required if name not in header]
        if absent:
            raise ValueError("Required columns are missing: %s" % ", ".join(absent))
        keep = [name for name in READ_COLUMNS if name in header]
        positions = [header.index(name) for name in keep]
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                bad_width += 1
                continue
            rows.append([row[pos].strip() for pos in positions])
    if bad_width:
        raise ValueError("CSV contains %d records with an unexpected column count; no rows were silently removed." % bad_width)
    if not rows:
        raise ValueError("Input CSV contains no data records.")
    return pd.DataFrame(rows, columns=keep), header


def _numeric_column(series, name):
    declared_missing = _missing(series)
    values = pd.to_numeric(series.mask(declared_missing), errors="coerce").astype(float)
    malformed = ~declared_missing & values.isna()
    if malformed.any():
        raise ValueError("Numeric column %s contains %d malformed, non-missing values; inspect source encoding." % (name, int(malformed.sum())))
    nonfinite = ~declared_missing & ~np.isfinite(values.to_numpy())
    values = values.mask(nonfinite)
    return values, {
        "declared_missing_n": int(declared_missing.sum()),
        "nonfinite_as_missing_n": int(nonfinite.sum()),
        "missing_total_n": int(values.isna().sum()),
        "malformed_numeric_n": 0,
    }


def _action_counts(raw):
    report = {"used_in_features": False, "used_in_simulated_propensity": False}
    codes = {}
    for name in OBSERVED_ACTIONS:
        if name not in raw:
            report[name] = {"available": False}
            continue
        missing = _missing(raw[name])
        numeric = pd.to_numeric(raw[name].mask(missing), errors="coerce")
        valid = numeric.isin([0, 1]) & ~missing
        codes[name] = numeric.where(valid)
        report[name] = {
            "available": True,
            "zero_n": int((numeric == 0).sum()),
            "one_n": int((numeric == 1).sum()),
            "missing_n": int(missing.sum()),
            "unrecognized_n": int((~valid & ~missing).sum()),
        }
    if all(name in codes for name in OBSERVED_ACTIONS):
        first, second = [codes[name] for name in OBSERVED_ACTIONS]
        report["joint_counts_statin_antihtn"] = {
            "%d%d" % (a, b): int(((first == a) & (second == b)).sum())
            for a in (0, 1) for b in (0, 1)
        }
        report["joint_unrecognized_or_missing_n"] = int((first.isna() | second.isna()).sum())
    return report


def load_cvd(input_path, member=None, max_population=None, population_seed=20260914):
    """Return (feature_only_dataframe, aggregate_report) after full validation.

    PATID is a trimmed string, with no numeric conversion.  Duplicate or missing
    IDs are rejected, including repeated IDs at different index dates.  Optional
    subsampling is a seeded random sample taken only after validating every row.
    No raw identifiers, dates, treatment flags, or patient records enter reports.
    """
    if max_population is not None and (int(max_population) != max_population or max_population <= 0):
        raise ValueError("max_population must be a positive integer or None.")
    source, zip_member = _resolve_source(input_path, member)
    raw, header = _read_selected(source, zip_member)
    missing_id = _missing(raw["PATID"])
    duplicate_id = raw.loc[~missing_id, "PATID"].duplicated(keep=False)
    if missing_id.any() or duplicate_id.any():
        raise ValueError("Single-index cohort validation failed: missing PATID rows=%d; rows in duplicated PATID groups=%d. IDs remain strings; repeated patients are not collapsed." % (int(missing_id.sum()), int(duplicate_id.sum())))
    date_report = {"available": "INDEX_DATE" in raw, "used_in_features": False}
    if "INDEX_DATE" in raw:
        missing_date = _missing(raw["INDEX_DATE"])
        dates = pd.to_datetime(raw["INDEX_DATE"].mask(missing_date), errors="coerce", utc=True)
        # Pandas 2 uses a common inferred format. Retry nonmissing failures
        # individually to accept mixed ISO / US formats without a version gate.
        retry = dates.isna() & ~missing_date
        if retry.any():
            reparsed = [pd.to_datetime(value, errors="coerce", utc=True) for value in raw.loc[retry, "INDEX_DATE"]]
            dates.loc[retry] = pd.Series(reparsed, index=raw.index[retry], dtype="datetime64[ns, UTC]")
        invalid_date = dates.isna() & ~missing_date
        date_report.update({"missing_n": int(missing_date.sum()), "unparsed_n": int(invalid_date.sum()), "ambiguous_numeric_format_assumption": "month-first"})
        if missing_date.any() or invalid_date.any():
            raise ValueError("INDEX_DATE validation failed: missing rows=%d; unparsed rows=%d." % (int(missing_date.sum()), int(invalid_date.sum())))
    frame = pd.DataFrame(index=raw.index)
    missing_report = {}
    for name in REQUIRED_NUMERIC + OPTIONAL_NUMERIC:
        if name in raw:
            frame[name], missing_report[name] = _numeric_column(raw[name], name)
            if name in REQUIRED_NUMERIC and frame[name].isna().all():
                raise ValueError("Required numeric column %s is entirely missing." % name)
    categorical_report = {}
    for name in CATEGORICAL:
        if name in raw:
            missing = _missing(raw[name])
            frame[name] = raw[name].mask(missing, None)
            categorical_report[name] = {"missing_n": int(missing.sum()), "nonmissing_distinct_n": int(raw.loc[~missing, name].nunique())}
    observed_actions = _action_counts(raw)
    full_n = len(frame)
    if max_population is not None and full_n > int(max_population):
        take = np.random.default_rng(population_seed).choice(full_n, size=int(max_population), replace=False)
        frame = frame.iloc[np.sort(take)].copy()
    frame = frame.reset_index(drop=True)
    report = {
        "input_path": str(source),
        "requested_member": member,
        "selected_zip_member": zip_member,
        "full_input_rows": int(full_n),
        "selected_population_rows": int(len(frame)),
        "population_seed": int(population_seed),
        "subsampling": "seeded random without replacement after full validation" if len(frame) < full_n else "none",
        "full_input_distinct_patid": int(full_n),
        "single_index_per_patient": True,
        "patid_handling": "whitespace-trimmed strings; no numeric conversion; discarded after validation",
        "index_date": date_report,
        "retained_baseline_columns": list(frame.columns),
        "absent_optional_columns": [name for name in OPTIONAL_NUMERIC + CATEGORICAL if name not in raw],
        "excluded_from_features": [name if name else "<unnamed_column>" for name in header if name not in frame.columns],
        "numeric_quality_full_input": missing_report,
        "categorical_quality_full_input": categorical_report,
        "missing_n_selected_population": {name: int(frame[name].isna().sum()) for name in frame.columns},
        "observed_action_aggregate_full_input": observed_actions,
        "declared_missing_tokens_case_insensitive": sorted(MISSING_TOKENS),
        "patient_records_exported": False,
        "baseline_timing": "candidate baseline fields; source timing and prior imputation are not confirmed",
    }
    return frame, report


def build_basis(df):
    """Return (finite float64 basis, standardized numeric dict, report).

    Numeric features: median imputation, empirical 0.5/99.5% clipping,
    standardization, linear / centered-square / pairwise interaction terms.
    Categorical features: ten most frequent nonmissing levels plus OTHER and
    MISSING when observed; one reference level omitted.  Ordinal level names in
    the report avoid exporting original category text.  No p-star, oracle, action,
    outcome, follow-up, or risk-score values are inspected here.
    """
    if len(df) < 2:
        raise ValueError("At least two covariate rows are required.")
    absent = [name for name in REQUIRED_NUMERIC if name not in df]
    if absent:
        raise ValueError("Missing required basis columns: %s" % ", ".join(absent))
    columns, names = [np.ones(len(df), dtype=np.float64)], ["intercept"]
    zdict, numeric_report, categorical_report, dropped = {}, {}, {}, []

    def add(name, values):
        vector = np.asarray(values, dtype=np.float64)
        centered = vector - vector.mean()
        scale = float(centered.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            dropped.append(name)
            return
        columns.append(centered / scale)
        names.append(name)

    for name in REQUIRED_NUMERIC + OPTIONAL_NUMERIC:
        if name not in df:
            continue
        values, quality = _numeric_column(df[name], name)
        missing = values.isna().to_numpy()
        if missing.all():
            if name in REQUIRED_NUMERIC:
                raise ValueError("Required numeric column %s is entirely missing in the selected population." % name)
            dropped.append(name + "[all_missing]")
            continue
        observed = values.to_numpy()[~missing]
        median = float(np.median(observed))
        low, high = np.quantile(observed, [0.005, 0.995])
        imputed = values.fillna(median).to_numpy(dtype=np.float64)
        clipped = np.clip(imputed, low, high)
        mean, sd = float(clipped.mean()), float(clipped.std(ddof=0))
        z = (clipped - mean) / sd if sd > 1e-12 else np.zeros(len(df))
        zdict[name] = z
        numeric_report[name] = {
            **quality,
            "median_imputation": median,
            "clip_quantiles": [0.005, 0.995],
            "clip_limits": [float(low), float(high)],
            "clipped_mean": mean,
            "clipped_sd": sd,
            "outside_clip_limits_n": int(((imputed < low) | (imputed > high)).sum()),
        }
        add(name + "[linear]", z)
        add(name + "[square]", z ** 2)
        if missing.any():
            add(name + "[missing]", missing)
    for i, first in enumerate(REQUIRED_NUMERIC):
        for second in REQUIRED_NUMERIC[i + 1:]:
            add(first + "*" + second, zdict[first] * zdict[second])
    for name in CATEGORICAL:
        if name not in df:
            continue
        series = df[name].astype(object)
        missing = _missing(series)
        clean = series.astype(str).str.strip()
        counts = Counter(clean.loc[~missing].tolist())
        levels = sorted(counts, key=lambda item: (-counts[item], item))[:10]
        level_index = {value: index for index, value in enumerate(levels)}
        # Numeric integer codes keep the OTHER/MISSING sentinels distinct from
        # any category literally named "OTHER" or "MISSING" in source data.
        other_code, missing_code = len(levels), len(levels) + 1
        encoded = np.array([missing_code if is_missing else level_index.get(value, other_code) for value, is_missing in zip(clean, missing)], dtype=np.int64)
        codes = list(range(len(levels)))
        if np.any(encoded == other_code):
            codes.append(other_code)
        if np.any(encoded == missing_code):
            codes.append(missing_code)
        labels = {code: ("missing" if code == missing_code else "other" if code == other_code else "level_%02d" % (code + 1)) for code in codes}
        reference = codes[0]
        categorical_report[name] = {
            "nonmissing_distinct_n": len(counts),
            "retained_nonmissing_level_n": len(levels),
            "selection": "top 10 nonmissing levels by frequency; lexical tie break",
            "reference_level": labels[reference],
            "level_counts": {labels[code]: int((encoded == code).sum()) for code in codes},
            "raw_category_values_exported": False,
        }
        dropped.append(name + "[" + labels[reference] + "][reference]")
        for code in codes[1:]:
            add(name + "[" + labels[code] + "]", encoded == code)
    x = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Basis construction produced a nonfinite value.")
    report = {
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "feature_names": names,
        "dropped_features": dropped,
        "numeric_preprocessing": numeric_report,
        "categorical_preprocessing": categorical_report,
        "final_nonintercept_scaling": "centered and population-standardized to unit SD; constants removed",
        "fit_scope": "entire fixed empirical covariate population",
        "interpretation": "transductive finite-population benchmark; not new-patient generalization",
        "oracle_or_propensity_used_in_basis": False,
        "observed_actions_or_outcomes_used_in_basis": False,
    }
    return x, zdict, report


def make_mock_population(n=4000, seed=20260914):
    """Create explicitly fictional covariates for software smoke tests only."""
    if int(n) != n or n < 2:
        raise ValueError("Mock population size must be an integer of at least 2.")
    n = int(n)
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(58, 11, n), 30, 85)
    bmi = np.clip(27 + 0.035 * (age - 58) + rng.normal(0, 4.5, n), 17, 48)
    sbp = np.clip(126 + 0.4 * (age - 58) + 0.8 * (bmi - 27) + rng.normal(0, 13, n), 85, 195)
    tc = np.clip(194 + 1.5 * (bmi - 27) + rng.normal(0, 31, n), 100, 320)
    hdlc = np.clip(54 - 0.7 * (bmi - 27) + rng.normal(0, 11, n), 20, 105)
    egfr = np.clip(91 - 0.65 * (age - 58) - 0.06 * (sbp - 126) + rng.normal(0, 14, n), 20, 135)
    sex = rng.choice(["0", "1"], n)
    diabetes = (rng.random(n) < 1 / (1 + np.exp(-(-1.7 + 0.08 * (bmi - 27) + 0.02 * (age - 58))))).astype(int)
    smoker = (rng.random(n) < 0.16).astype(int)
    antihtn = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.05 * (sbp - 126))))).astype(int)
    statin = (rng.random(n) < 1 / (1 + np.exp(-(-1.1 + 0.04 * (age - 58) + 0.012 * (tc - 194) + 0.4 * diabetes)))).astype(int)
    return pd.DataFrame({
        "PATID": ["MOCK_%08d" % i for i in range(n)],
        "INDEX_DATE": ["2020-01-01"] * n,
        "age": age, "sbp": sbp, "bmi": bmi, "tc": tc, "hdlc": hdlc, "eGFR": egfr,
        "ADI_NATRANK": rng.integers(1, 101, n),
        "sex": sex,
        "race": rng.choice(["mock_group_A", "mock_group_B", "mock_group_C", "mock_group_D"], n, p=[0.45, 0.3, 0.15, 0.1]),
        "diabetes": diabetes,
        "smoker": smoker,
        "insurance_type": rng.choice(["mock_plan_A", "mock_plan_B", "mock_plan_C"], n, p=[0.4, 0.35, 0.25]),
        "statin": statin, "antihtn": antihtn,
    })
