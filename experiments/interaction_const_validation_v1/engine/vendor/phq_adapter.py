"""Load the archived PHQ-9 semi-synthetic design without changing its population.

Only the five original design inputs are required. No archived oracle Gram,
moment, projection, clinical outcome, or observed treatment file is read.
The original query library is deliberately reused; its held-out split is not
a fresh query holdout for this development comparison.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading

import numpy as np
import pandas as pd

try:
    from .frozen import phq_backend as core
except ImportError:
    from frozen import phq_backend as core


DEFAULT_ROOT = Path.home() / "qaoct_semisynthetic_v1"
INPUT_FILES = {
    "history": "history_bank_v1_2/history_bank_v1_2.csv.gz",
    "propensity": "propensity_generator_v1/propensity_scores_oof_v1.csv.gz",
    "library": "query_library_v1/query_library_v1.csv",
    "basis": "sieve_v1_1/history_basis_X_v1_1.npy",
    "index": "sieve_v1_1/history_basis_row_index_v1_1.csv.gz",
}
_MOCK_IMPORT_LOCK = threading.RLock()


def _read_table(path, name, identifier=None):
    try:
        return pd.read_csv(path, dtype={identifier: str} if identifier else None,
                           keep_default_na=False, low_memory=False)
    except Exception:
        # Parser errors can quote source records; expose only the input role.
        raise ValueError(f"Unable to read PHQ-9 {name} input table.") from None


def _integers(values, label, lower, upper=None):
    number = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(number) & (number == np.floor(number)) & (number >= lower)
    if upper is not None:
        valid &= number <= upper
    if not valid.all():
        raise ValueError(f"PHQ-9 {label} must contain valid integral values.")
    return number.astype(np.int64)


def _validate_keys(frame, name):
    if not {core.ID_COL, core.STEP_COL}.issubset(frame.columns):
        raise ValueError(f"PHQ-9 {name} is missing patient-step key columns.")
    if frame.empty or frame[core.ID_COL].isna().any():
        raise ValueError(f"PHQ-9 {name} has missing patient identifiers or no rows.")
    frame[core.ID_COL] = frame[core.ID_COL].astype(str)
    if frame[core.ID_COL].str.strip().eq("").any():
        raise ValueError(f"PHQ-9 {name} has blank patient identifiers.")
    frame[core.STEP_COL] = _integers(frame[core.STEP_COL], f"{name} steps", 0, 3)
    if frame.duplicated([core.ID_COL, core.STEP_COL]).any():
        raise ValueError(f"PHQ-9 {name} has duplicate patient-step rows.")
    if not frame.groupby(core.ID_COL, sort=False).size().eq(4).all():
        raise ValueError(f"PHQ-9 {name} must contain exactly steps 0, 1, 2, 3 per patient.")
    return pd.MultiIndex.from_frame(frame[[core.ID_COL, core.STEP_COL]])


def _numeric_matrix(frame, columns, label):
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"PHQ-9 {label} is missing required numeric columns.")
    matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"PHQ-9 {label} contains nonfinite or nonnumeric values.")
    return matrix


def load_real(root=DEFAULT_ROOT):
    """Return a data dict; archived p_star is used unchanged as the simulator."""
    root = Path(root).expanduser()
    paths = {name: root / relative for name, relative in INPUT_FILES.items()}
    missing = [INPUT_FILES[name] for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PHQ-9 design input files: " + ", ".join(missing))

    history = _read_table(paths["history"], "history", core.ID_COL)
    propensity = _read_table(paths["propensity"], "propensity", core.ID_COL)
    index = _read_table(paths["index"], "basis row index", core.ID_COL)
    library = _read_table(paths["library"], "query library", "query_id")
    key_sets = [_validate_keys(frame, name) for name, frame in
                (("history", history), ("propensity", propensity), ("basis row index", index))]
    if any(len(keys) != len(key_sets[0]) or len(keys.difference(key_sets[0]))
           for keys in key_sets[1:]):
        raise ValueError("PHQ-9 history, propensity and basis index patient-step sets differ.")
    if "row_number" not in index:
        raise ValueError("PHQ-9 basis index is missing row_number.")
    index["row_number"] = _integers(index["row_number"], "basis row numbers", 0, len(index) - 1)
    if not np.array_equal(np.sort(index["row_number"].to_numpy()), np.arange(len(index))):
        raise ValueError("PHQ-9 basis row numbers must be unique and contiguous from zero.")

    try:
        x = np.load(paths["basis"], mmap_mode="r", allow_pickle=False)
    except Exception:
        raise ValueError("Unable to read the PHQ-9 numeric basis array.") from None
    if x.ndim != 2 or x.shape[0] != len(index) or x.shape[1] < 1 or x.dtype.kind not in "fiu":
        raise ValueError("PHQ-9 basis must be a real numeric matrix aligned to the complete row index.")
    if not np.isfinite(x).all() or not np.allclose(x[:, 0], 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("PHQ-9 basis must be finite and have an initial intercept column of ones.")
    probability_columns = [f"p_star__{a}" for a in core.ACTION_ORDER]
    p_star = _numeric_matrix(propensity, probability_columns, "simulator probabilities")
    if np.any(p_star < 0.01 - 1e-10) or np.any(p_star > 1) or not np.allclose(
            p_star.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("PHQ-9 simulator probabilities must sum to one and retain the original 0.01 floor.")
    propensity[probability_columns] = p_star
    descriptors = core.validate_query_library(library)
    if library["query_id"].isna().any() or library["query_id"].astype(str).str.strip().eq("").any():
        raise ValueError("PHQ-9 query identifiers cannot be missing or blank.")
    _numeric_matrix(library, [f"h_{t}" for t in range(4)] +
                    [f"c__{a}" for a in core.ACTION_ORDER] +
                    [f"eta_{j}" for j in range(3)] + descriptors, "query definitions")
    # The frozen alignment selects the original pre-action history columns,
    # and the frozen tilt builder preserves its original missing-value rules.
    population = core.align_population(history, propensity, index, x)
    queries = core.prepare_queries(library, descriptors,
        population[core.STEP_COL].to_numpy(dtype=int), core.build_tilt_features(population))
    metadata = {
        "dataset": "PHQ-9", "mode": "semi_synthetic", "population_rows": len(population),
        "patients": int(population[core.ID_COL].nunique()), "basis_dimension": int(x.shape[1]),
        "step_count": 4, "action_count": 4, "query_count": len(library),
        "query_split_counts": {k: int(v) for k, v in library["query_split"].value_counts().items()},
        "archived_query_library_reused": True, "fresh_query_holdout": False,
        "clinical_actions_used": False, "clinical_outcomes_used_as_targets": False,
        "simulator_probabilities_changed": False, "archived_oracle_arrays_required": False,
        "sampling_unit": "patient_step_row_with_replacement",
        "files": {name: {"relative_path": INPUT_FILES[name]} for name in paths},
    }
    return dict(core=core, population=population, x_population=x, queries=queries,
                query_library=library, input_metadata=metadata)


def load_mock(patients=1000, seed=2026091701):
    """Use the frozen artificial fixture; this branch never opens real inputs."""
    if isinstance(patients, bool) or int(patients) != patients or patients < 2:
        raise ValueError("PHQ-9 mock requires an integral patient count of at least two.")
    source = Path(__file__).resolve().parent / "frozen" / "mock_data.py"
    spec = importlib.util.spec_from_file_location("_semi_propensity_phq_mock", source)
    module = importlib.util.module_from_spec(spec)
    # The original file imports qaoct_backend by its archived top-level name.
    # Bind that name only while importing and restore any caller's binding.
    with _MOCK_IMPORT_LOCK:
        sentinel = object()
        previous = sys.modules.get("qaoct_backend", sentinel)
        sys.modules["qaoct_backend"] = core
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is sentinel:
                sys.modules.pop("qaoct_backend", None)
            else:
                sys.modules["qaoct_backend"] = previous
    data = module.make_mock(patients=int(patients), seed=int(seed))
    # The fixture computes references internally. Discard them so the runner's
    # post-fit evaluation path is the same as for real archived covariates.
    data = {name: data[name] for name in ("population", "x_population", "queries", "query_library")}
    data["core"] = core
    data["input_metadata"] = {
        "dataset": "PHQ-9", "mode": "mock", "population_rows": 4 * int(patients),
        "patients": int(patients), "basis_dimension": int(data["x_population"].shape[1]),
        "step_count": 4, "action_count": 4, "query_count": len(data["query_library"]),
        "query_split_counts": {k: int(v) for k, v in data["query_library"]["query_split"].value_counts().items()},
        "mock_seed": int(seed), "real_patient_data_used": False,
        "archived_query_library_reused": False,
        "fresh_query_holdout": False,
        "query_holdout_note": "Artificial query split; tilt components overlap across splits.",
        "sampling_unit": "patient_step_row_with_replacement",
    }
    return data
