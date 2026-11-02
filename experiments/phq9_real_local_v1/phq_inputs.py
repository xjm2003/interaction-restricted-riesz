"""Read original local real-outcome files; never query a database or simulate Y."""
from pathlib import Path
import csv
import gzip
import re
import numpy as np
import pandas as pd

ACTIONS = ('no_active_treatment', 'antidepressant_only', 'psychotherapy_only', 'combined')
MISSING = {'', 'na', 'nan', 'none', 'null', '<na>', 'nat'}
DEFAULT_FEATURES = ('baseline_phq9', 'age_at_index', 'sex', 'race', 'hispanic')
FORBIDDEN = re.compile(r'(followup|follow_up|outcome|phq9_90d|phq9_change|remission|response|observed_next|m_star|p_star|action|treatment|interval|death|last_phq|mean_phq|min_phq|max_phq|n_phq|days_|step|index_date|patid|^id$)', re.I)


class InputError(ValueError):
    """Safe messages omit individual records, values and dates."""


def locate(root, explicit, filename, required=True):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise InputError('Explicit input file is absent: ' + str(path))
        return path
    roots = [root, root / 'qaoct_semisynthetic_v1']
    found = list(dict.fromkeys((r / n).resolve() for r in roots
                              for n in (filename, filename + '.gz') if (r / n).is_file()))
    if not found:
        if required:
            raise InputError('Cannot find ' + filename + ' under --data-root. Specify its existing local path; no database connection is required.')
        return None
    if len(found) > 1:
        raise InputError('Multiple files match ' + filename + '; select one explicitly. No version is selected by its results.')
    return found[0]


def read_table(path, role):
    try:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream))
        names = [str(c).strip().lower() for c in header]
        if len(names) != len(set(names)) or any(not c for c in names):
            raise InputError(role + ': empty or duplicate raw CSV column names.')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    except InputError:
        raise
    except Exception:
        raise InputError('Cannot parse ' + role + ' CSV; source values are not printed.') from None
    columns = [str(c).strip().lower() for c in frame.columns]
    if len(columns) != len(set(columns)):
        raise InputError(role + ': duplicate column names after case normalization.')
    frame.columns = columns
    if frame.empty:
        raise InputError(role + ': empty file.')
    return frame


def need(frame, columns, role):
    absent = set(columns) - set(frame.columns)
    if absent:
        raise InputError(role + ': missing columns: ' + ', '.join(sorted(absent)))


def missing(series):
    return series.astype(str).str.strip().str.lower().isin(MISSING)


def numeric(series, name, allow_missing=False):
    is_missing = missing(series)
    values = pd.to_numeric(series.where(~is_missing, np.nan), errors='coerce')
    bad = (~is_missing & ~np.isfinite(values)) | ((not allow_missing) & is_missing)
    if bad.any():
        raise InputError(name + ': missing or invalid numeric values; no silent replacement.')
    return values.astype(float)


def dates(series, name):
    # Existing exports use ISO dates, optionally followed by a time component.
    date_part = series.astype(str).str.strip().str.extract(r'^(\d{4}-\d{2}-\d{2})(?:$|[ T])', expand=False)
    result = pd.to_datetime(date_part, format='%Y-%m-%d', errors='coerce')
    if result.isna().any():
        raise InputError(name + ': expected nonmissing ISO calendar dates.')
    return result


def keys(frame, role):
    need(frame, ['patid'], role)
    frame = frame.copy()
    frame['patid'] = frame.patid.astype(str).str.strip()
    if missing(frame.patid).any():
        raise InputError(role + ': missing patient identifiers.')
    if frame.patid.duplicated().any():
        raise InputError(role + ': multiple records per PATID. Resolve episode keys explicitly; no arbitrary deduplication.')
    return frame


def join_verified(base, other, role, required):
    """Preserve selected outcome rows and verify all shared episode information."""
    other = keys(other, role).set_index('patid')
    aligned = other.reindex(base.patid)
    aligned.index = base.index
    matched = base.patid.isin(other.index)
    if required and not matched.all():
        raise InputError(role + ': some outcome patients have no matching input row. Missing actions are never filled as untreated.')
    if 'index_date' in other:
        check = dates(aligned.loc[matched, 'index_date'], role + ' index_date')
        if not np.array_equal(check.to_numpy(), base.loc[matched, 'index_date'].to_numpy()):
            raise InputError(role + ': index dates disagree with the outcome table.')
    if 'baseline_phq9' in other:
        check = numeric(aligned.loc[matched, 'baseline_phq9'], role + ' baseline_phq9')
        if not np.allclose(check.to_numpy(), base.loc[matched, 'baseline_phq9'].to_numpy(), atol=1e-8, rtol=0):
            raise InputError(role + ': baseline PHQ-9 disagrees with the outcome table.')
    return aligned, matched


def load_inputs(data_root, outcome=None, actions=None, baseline=None, index=None, covariates=None):
    root = Path(data_root).expanduser().resolve()
    paths = {
        'outcome': locate(root, outcome, 'phq9_90d_analytic.csv'),
        'actions': locate(root, actions, 'phq9_gnet_panel_with_actions.csv'),
        'baseline': locate(root, baseline, 'phq9_index_with_demo.csv', required=False),
        'index': locate(root, index, 'phq9_index_cohort.csv', required=False),
    }
    ytable = keys(read_table(paths['outcome'], 'outcome'), 'outcome')
    need(ytable, ['index_date', 'baseline_phq9', 'followup_date', 'phq9_90d'], 'outcome')
    ytable['index_date'] = dates(ytable.index_date, 'outcome index_date')
    ytable['followup_date'] = dates(ytable.followup_date, 'outcome followup_date')
    ytable['baseline_phq9'] = numeric(ytable.baseline_phq9, 'baseline_phq9')
    y = numeric(ytable.phq9_90d, 'phq9_90d').to_numpy()
    if ((y < 0) | (y > 27)).any() or not ytable.baseline_phq9.between(0, 27).all():
        raise InputError('PHQ-9 values must be between 0 and 27; no outcome clipping is performed.')
    followup = (ytable.followup_date - ytable.index_date).dt.days
    if not followup.between(60, 120).all():
        raise InputError('Observed outcome dates fall outside the original inclusive 60..120 day window. No rows are silently removed.')
    if 'days_to_followup' in ytable:
        if not np.allclose(numeric(ytable.days_to_followup, 'days_to_followup'), followup, atol=1e-8, rtol=0):
            raise InputError('Stored follow-up days disagree with the actual dates.')

    panel = read_table(paths['actions'], 'actions')
    need(panel, ['patid', 'step'], 'actions')
    step = numeric(panel.step, 'action panel step')
    if not np.equal(step, np.floor(step)).all() or not step.isin([0, 1, 2, 3]).all():
        raise InputError('Action panel must use steps 0,1,2,3; no index shift is inferred.')
    panel = panel.loc[step.eq(0)].copy()
    if panel.empty:
        raise InputError('Action panel contains no step-0 records.')
    col = 'action_t' if 'action_t' in panel else 'action'
    need(panel, [col], 'actions')
    panel['_chosen_action'] = panel[col].str.strip().str.lower()
    if not panel._chosen_action.isin(ACTIONS).all():
        raise InputError('Step-0 action labels are missing or outside the four original named categories.')
    if 'action_t' in panel and 'action' in panel:
        if not panel.action_t.str.strip().str.lower().eq(panel.action.str.strip().str.lower()).all():
            raise InputError('action_t and action disagree; choose/reconcile the source before fitting.')
    if {'has_antidepressant', 'has_psychotherapy'}.issubset(panel.columns):
        ad = numeric(panel.has_antidepressant, 'has_antidepressant')
        psy = numeric(panel.has_psychotherapy, 'has_psychotherapy')
        if not ad.isin([0, 1]).all() or not psy.isin([0, 1]).all():
            raise InputError('Action component indicators must be zero or one.')
        expected = np.asarray(ACTIONS)[(ad + 2 * psy).to_numpy(int)]
        if not np.array_equal(expected, panel._chosen_action.to_numpy()):
            raise InputError('Action labels disagree with antidepressant/psychotherapy indicators.')
    linked, _ = join_verified(ytable, panel, 'step-0 actions', required=True)
    action = linked._chosen_action.map({name: i for i, name in enumerate(ACTIONS)}).to_numpy(int)
    timing = {'step_0_selected': True, 'index_date_verified': 'index_date' in panel}
    for column, expected in [('interval_start', 0), ('interval_end', 30)]:
        timing[column + '_verified'] = column in linked
        if column in linked:
            actual = (dates(linked[column], 'actions ' + column) - ytable.index_date).dt.days
            if not actual.eq(expected).all():
                raise InputError('Step-0 ' + column + ' is inconsistent with the original [0,30) window; no automatic realignment.')
    warnings = [
        'Selected observed-outcome cohort: selection/transport to the full index population is not identified by this code.',
        'Exposure is recorded treatment during index days [0,30), not baseline initiation, adherence, or a sustained treatment regime.',
        'No recorded active treatment is a source-record category, not verified absence of treatment.',
        'Pre-index confounding sufficiency, measurement timing, eligibility, consistency and positivity need substantive assessment.',
        'Interaction-restricted correction adds a structural assumption; it has no general unrestricted AIPW guarantee.',
    ]
    if not all(timing.values()):
        warnings.append('Some action dates are unavailable: step-0 timing is inherited from the named source and not independently verified.')
    if (ytable.baseline_phq9 < 10).any():
        warnings.append('Some supplied baseline scores are below 10; no eligibility filter was silently imposed.')

    feature_source = pd.DataFrame({'baseline_phq9': ytable.baseline_phq9})
    baseline_match_count = 0
    if paths['baseline']:
        b = read_table(paths['baseline'], 'baseline')
        linked_b, matched_b = join_verified(ytable, b, 'baseline', required=True)
        baseline_match_count = int(matched_b.sum())
        for c in linked_b:
            if c not in feature_source and c not in ('patid', 'index_date'):
                feature_source[c] = linked_b[c]
        if 'index_date' not in b:
            # Recorded demographics may be static, but an age-at-index cannot be verified this way.
            if 'age_at_index' in feature_source:
                feature_source = feature_source.drop(columns=['age_at_index'])
            warnings.append('Baseline table has no index_date: age_at_index excluded; demographic timing remains unverified.')
    else:
        warnings.append('No baseline demographic table found: only baseline_phq9 is available by default; confounding adjustment is especially limited.')
    selected = ([c.strip().lower() for c in covariates.split(',') if c.strip()]
                if covariates else [c for c in DEFAULT_FEATURES if c in feature_source])
    if 'baseline_phq9' not in selected:
        selected.insert(0, 'baseline_phq9')
    if len(selected) != len(set(selected)):
        raise InputError('Duplicate requested covariate names.')
    for c in selected:
        if c != 'baseline_phq9' and FORBIDDEN.search(c):
            raise InputError('Disallowed identifier/future/outcome/treatment covariate: ' + c)
        if c not in feature_source:
            raise InputError('Requested baseline covariate is absent: ' + c)
    X = feature_source[selected].copy()
    feature_audit = []
    for c in selected:
        if c == 'baseline_phq9':
            X[c] = ytable.baseline_phq9.astype(float)
        else:
            s = X[c].astype(str).str.strip()
            m = missing(s)
            numbers = pd.to_numeric(s.where(~m, np.nan), errors='coerce')
            forced_numeric = c == 'age_at_index'
            if forced_numeric or (c not in ('sex', 'race', 'hispanic') and not s[~m].empty and numbers[~m].notna().all()):
                X[c] = numeric(s, 'baseline covariate ' + c, allow_missing=True)
            else:
                if s[~m].nunique() > 100:
                    raise InputError('Baseline categorical covariate has over 100 distinct levels: ' + c)
                X[c] = s.where(~m, np.nan).astype(object)
        if c == 'age_at_index' and not X[c].dropna().between(0, 120).all():
            raise InputError('age_at_index contains out-of-range values.')
        feature_audit.append({'name': c, 'missing_count': int(X[c].isna().sum()),
                              'kind': 'numeric' if pd.api.types.is_numeric_dtype(X[c]) else 'categorical'})
    if covariates:
        warnings.append('Custom covariates were explicitly requested; names alone cannot certify that they were measured before treatment.')
    candidate = None
    if paths['index']:
        idx = keys(read_table(paths['index'], 'index'), 'index')
        join_verified(ytable, idx, 'index', required=True)
        candidate = {'candidate_rows': len(idx), 'outcome_rows_matched': len(ytable),
                     'candidate_rows_not_in_outcome_file': len(idx) - len(ytable),
                     'unmatched_interpretation': 'not present in supplied selected-outcome file; not established missing PHQ or loss to follow-up'}
    audit = {
        'analysis_status': 'EXPLORATORY_SELECTED_COHORT',
        'data_type': 'observed clinical PHQ-9, no simulated outcome or treatment',
        'input_files': {role: {'path': str(p)} for role, p in paths.items() if p},
        'n_patients': len(ytable), 'n_rows_per_patient': 1,
        'action_counts': {name: int((action == a).sum()) for a, name in enumerate(ACTIONS)},
        'observed_phq9_mean': float(np.mean(y)), 'observed_phq9_sd': float(np.std(y, ddof=1)),
        'followup_days_min': int(followup.min()), 'followup_days_max': int(followup.max()),
        'followup_days_median': float(followup.median()),
        'outcome_definition': 'supplied original selected PHQ-9 measurement in inclusive index day 60..120; not exact day 90',
        'baseline_matched_rows': baseline_match_count, 'features': feature_audit,
        'action_timing': timing, 'candidate_index_inventory': candidate,
        'selection_weights_fitted': False, 'clinical_record_retrieval_performed': False,
        'individual_records_written': False, 'warnings': warnings,
    }
    # Stable patient ordering ensures row order in exports does not change fold assignment.
    order = np.argsort(ytable.patid.to_numpy(), kind='stable')
    return X.iloc[order].reset_index(drop=True), action[order], y[order], audit
