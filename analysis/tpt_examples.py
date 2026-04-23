"""Reusable helpers for the TpT Computo paper examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import _tree, export_text

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.trees import (
    TpTDecisionTreeClassifier,
    TpTDecisionTreeRegressor,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

PBCSEQ_PATH = DATA_DIR / "pbcseq_classification_long.csv"
PRODUC_PATH = DATA_DIR / "produc_regression_long.csv"

PBCSEQ_FEATURES = ["bili", "albumin", "protime", "ast"]
PRODUC_FEATURES = ["pc", "emp", "unemp"]
PBCSEQ_MIN_VISITS = 2
PBCSEQ_MAX_VISITS = 4


def gamma_grid(max_gamma: float, step: float = 0.005) -> tuple[float, ...]:
    """Build a regular gamma grid including the upper bound."""

    n_steps = int(round(max_gamma / step))
    return tuple(round(step * idx, 6) for idx in range(n_steps + 1))


DEFAULT_CLASSIFICATION_GAMMAS = gamma_grid(0.4, step=0.001)
DEFAULT_REGRESSION_GAMMAS = gamma_grid(0.1, step=0.001)


@dataclass
class ExampleResult:
    """Container for a rendered example result."""

    metrics: dict[str, Any]
    predictions: pd.DataFrame
    tree_text: str
    model: Any
    feature_names: list[str]
    gamma_table: pd.DataFrame
    hyperparameters: pd.DataFrame
    gamma_profile: pd.DataFrame | None = None
    long_preview: pd.DataFrame | None = None
    wide_preview: pd.DataFrame | None = None
    stage_predictions: pd.DataFrame | None = None


def load_pbcseq_classification_data() -> pd.DataFrame:
    """Load the processed pbcseq classification dataset."""

    return pd.read_csv(PBCSEQ_PATH)


def load_produc_regression_data() -> pd.DataFrame:
    """Load the processed Produc regression dataset."""

    return pd.read_csv(PRODUC_PATH)


def build_pbcseq_visit_order_cohort(
    *,
    min_visits: int = PBCSEQ_MIN_VISITS,
    max_visits: int = PBCSEQ_MAX_VISITS,
) -> pd.DataFrame:
    """Build a variable-length pbcseq cohort using visit-order waves."""

    df = load_pbcseq_classification_data().sort_values(["id", "day"], kind="mergesort").copy()
    df["visit_index"] = df.groupby("id").cumcount()

    counts = df.groupby("id").size()
    keep_ids = counts[counts >= min_visits].index

    cohort = df[df["id"].isin(keep_ids) & df["visit_index"].lt(max_visits)].copy()
    cohort = cohort.sort_values(["id", "visit_index"], kind="mergesort").reset_index(drop=True)
    return cohort


def long_to_wide_frame(
    frame: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    longitudinal_columns: list[str],
    static_columns: list[str] | None = None,
    wave_format: str = "{feature}_t{wave}",
) -> tuple[pd.DataFrame, list[list[int]], list[int]]:
    """Reshape a long dataframe with the public LongitudinalDataset API."""

    static_columns = static_columns or []
    keep_columns = [id_col, time_col, *static_columns, *longitudinal_columns]
    dataset = LongitudinalDataset(file_path=None, data_frame=frame.loc[:, keep_columns].copy())
    wide = dataset.to_wide(
        id_col=id_col,
        time_col=time_col,
        longitudinal_columns=longitudinal_columns,
        static_columns=static_columns,
        wave_format=wave_format,
    )
    return wide, dataset.feature_groups(), dataset.non_longitudinal_features()


def dataset_summary_table() -> pd.DataFrame:
    """Return a compact summary of the datasets used in the manuscript."""

    pbcseq = build_pbcseq_visit_order_cohort()

    produc = load_produc_regression_data()
    produc_waves = produc.groupby("state")["time_index"].nunique()

    return pd.DataFrame(
        [
            {
                "dataset": "pbcseq",
                "unit": f"{pbcseq['id'].nunique()} subjects",
                "waves": f"{PBCSEQ_MIN_VISITS}-{PBCSEQ_MAX_VISITS} visit-order waves",
                "predictors": ", ".join(PBCSEQ_FEATURES),
                "target": "end-of-follow-up event",
            },
            {
                "dataset": "Produc",
                "unit": f"{produc['state'].nunique()} states",
                "waves": f"{int(produc_waves.iloc[0]) + 1} annual waves",
                "predictors": ", ".join(PRODUC_FEATURES),
                "target": "held-out final-year GSP",
            },
        ]
    )


def metrics_table(metrics: dict[str, Any], *, task: str) -> pd.DataFrame:
    """Render narrow metric tables suitable for the paper layout."""

    if task == "classification":
        order = [
            "train_subjects",
            "test_subjects",
            "selected_gamma",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "tree_total_impurity",
            "tree_mean_impurity",
            "tree_depth",
            "n_nodes",
            "duration_nodes",
            "duration_sample_weight",
        ]
        labels = {
            "train_subjects": "Training subjects",
            "test_subjects": "Test subjects",
            "selected_gamma": "Selected gamma",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced accuracy",
            "roc_auc": "ROC AUC",
            "tree_total_impurity": "Tree total impurity",
            "tree_mean_impurity": "Tree mean impurity",
            "tree_depth": "Tree depth",
            "n_nodes": "Number of nodes",
            "duration_nodes": "Duration leaves",
            "duration_sample_weight": "Duration leaf sample weight",
        }
    elif task == "regression":
        order = [
            "train_subjects",
            "test_subjects",
            "selected_gamma",
            "mae",
            "r2",
            "tree_total_impurity",
            "tree_mean_impurity",
            "tree_depth",
            "n_nodes",
        ]
        labels = {
            "train_subjects": "Training states",
            "test_subjects": "Test states",
            "selected_gamma": "Selected gamma",
            "mae": "MAE",
            "r2": "R²",
            "tree_total_impurity": "Tree total impurity",
            "tree_mean_impurity": "Tree mean impurity",
            "tree_depth": "Tree depth",
            "n_nodes": "Number of nodes",
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

    def format_value(value: Any) -> str:
        if isinstance(value, (int, np.integer)):
            return f"{int(value)}"
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "nan"
            if abs(float(value)) >= 1000:
                return f"{float(value):,.3f}"
            return f"{float(value):.3f}"
        return str(value)

    rows = [{"metric": labels[key], "value": format_value(metrics[key])} for key in order]
    return pd.DataFrame(rows)


def summarize_tpt_tree(tree_) -> dict[str, float]:
    """Compute simple impurity summaries for a fitted TpT tree."""

    children_left = tree_.children_left
    leaf_mask = children_left == _tree.TREE_LEAF

    weighted_samples = np.asarray(tree_.weighted_n_node_samples, dtype=float)
    impurities = np.asarray(tree_.impurity, dtype=float)
    duration_mass = np.asarray(
        getattr(tree_, "weighted_n_duration", np.zeros_like(weighted_samples)),
        dtype=float,
    )
    duration_impurities = np.asarray(
        getattr(tree_, "impurity_duration", np.full_like(duration_mass, np.inf)),
        dtype=float,
    )

    duration_mask = (duration_mass > 0.0) & np.isfinite(duration_impurities)
    duration_node_count = int(np.sum(duration_mask))
    duration_sample_weight = float(np.sum(duration_mass[duration_mask]))

    impurity_sum = 0.0
    if np.any(leaf_mask):
        impurity_sum += float(np.dot(impurities[leaf_mask], weighted_samples[leaf_mask]))
    if np.any(duration_mask):
        impurity_sum += float(np.dot(duration_impurities[duration_mask], duration_mass[duration_mask]))

    total_weight = float(np.sum(weighted_samples[leaf_mask]) + np.sum(duration_mass[duration_mask]))
    mean_impurity = impurity_sum / total_weight if total_weight > 0.0 else float("nan")

    return {
        "tree_total_impurity": impurity_sum,
        "tree_mean_impurity": mean_impurity,
        "n_nodes": int(tree_.node_count),
        "tree_depth": int(tree_.max_depth),
        "duration_nodes": duration_node_count,
        "duration_sample_weight": duration_sample_weight,
    }


def tpt_tree_time_summary(model: Any) -> dict[str, int]:
    """Summarize selected wave indices in a fitted TpT tree."""

    tree_ = model.tree_
    split_times = np.asarray(
        getattr(tree_, "split_time_index", np.full(tree_.node_count, -1)),
        dtype=int,
    )
    internal_mask = tree_.feature != _tree.TREE_UNDEFINED
    internal_times = split_times[internal_mask]
    if internal_times.size == 0:
        return {"root_time": -1, "min_split_time": -1, "max_split_time": -1}
    return {
        "root_time": int(internal_times[0]),
        "min_split_time": int(internal_times.min()),
        "max_split_time": int(internal_times.max()),
    }


def select_gamma_by_tree_total_impurity(
    gamma_grid: tuple[float, ...],
    estimator_factory: Callable[[float], Any],
    X_train: Any,
    y_train: Any,
) -> tuple[float, Any, pd.DataFrame]:
    """Select the gamma that minimizes weighted total leaf impurity."""

    fitted_models: dict[float, Any] = {}
    records: list[dict[str, Any]] = []

    for gamma in gamma_grid:
        model = estimator_factory(float(gamma))
        model.fit(X_train, y_train)
        fitted_models[float(gamma)] = model

        stats = summarize_tpt_tree(model.tree_)
        records.append(
            {
                "gamma": float(gamma),
                "tree_total_impurity": stats["tree_total_impurity"],
                "tree_mean_impurity": stats["tree_mean_impurity"],
                "tree_depth": stats["tree_depth"],
                "n_nodes": stats["n_nodes"],
            }
        )

    table = pd.DataFrame(records).sort_values("gamma", kind="mergesort").reset_index(drop=True)
    best_row = (
        table.sort_values(
            ["tree_total_impurity", "tree_mean_impurity", "gamma"],
            kind="mergesort",
        )
        .iloc[0]
    )
    best_gamma = float(best_row["gamma"])
    return best_gamma, fitted_models[best_gamma], table


def deduplicate_gamma_table(table: pd.DataFrame, *, decimals: int = 10) -> pd.DataFrame:
    """Keep only the lowest gamma for each distinct impurity value."""

    compressed = table.copy()
    compressed["_impurity_key"] = compressed["tree_total_impurity"].round(decimals)
    compressed = (
        compressed.sort_values(["_impurity_key", "gamma"], kind="mergesort")
        .drop_duplicates(subset="_impurity_key", keep="first")
        .drop(columns="_impurity_key")
        .sort_values("gamma", kind="mergesort")
        .reset_index(drop=True)
    )
    return compressed


def compress_regression_gamma_table(table: pd.DataFrame, *, decimals: int = 10) -> pd.DataFrame:
    """Keep only breakpoints where impurity or selected times change."""

    columns = ["tree_total_impurity", "root_time", "min_split_time", "max_split_time"]
    compressed = table.copy()
    for column in columns:
        compressed[f"_{column}_key"] = compressed[column].round(decimals)

    key_columns = [f"_{column}_key" for column in columns]
    keep_mask = compressed[key_columns].ne(compressed[key_columns].shift()).any(axis=1)
    compressed = compressed.loc[keep_mask].copy()
    compressed = compressed.drop(columns=key_columns).reset_index(drop=True)
    return compressed


def hyperparameter_table(params: dict[str, Any]) -> pd.DataFrame:
    """Render a compact hyperparameter table for the manuscript."""

    rows = [{"parameter": key, "value": value} for key, value in params.items()]
    return pd.DataFrame(rows)


def _time_index_from_name(feature_name: str) -> int:
    if "_t" not in feature_name:
        return 0
    return int(feature_name.rsplit("_t", maxsplit=1)[1])


def stagewise_history_predictions(
    model: Any,
    X: pd.DataFrame,
    observed: pd.Series,
) -> pd.DataFrame:
    """Predict the final outcome from baseline, mid-history, and full-history inputs."""

    wave_map = {_time_index_from_name(column): column for column in X.columns}
    max_wave = max(wave_map) if wave_map else 0
    global_mid_wave = max_wave // 2
    stages = [
        ("Baseline history", 0),
        ("Mid history", global_mid_wave),
        ("Full history", max_wave),
    ]

    records: list[dict[str, Any]] = []
    for stage_label, cutoff in stages:
        X_stage = X.copy()
        drop_columns = [column for column in X.columns if _time_index_from_name(column) > cutoff]
        if drop_columns:
            X_stage.loc[:, drop_columns] = np.nan
        predicted = model.predict(X_stage)
        for subject_id, observed_value, predicted_value in zip(observed.index, observed.to_numpy(), predicted):
            records.append(
                {
                    "subject_id": subject_id,
                    "stage": stage_label,
                    "observed": float(observed_value),
                    "predicted": float(predicted_value),
                }
            )
    return pd.DataFrame(records)


def run_classification_example(
    *,
    gamma_grid: tuple[float, ...] = DEFAULT_CLASSIFICATION_GAMMAS,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ExampleResult:
    """Fit and evaluate the pbcseq classification example."""

    df = build_pbcseq_visit_order_cohort()

    subject_targets = df.groupby("id")["event"].first()
    train_subjects, test_subjects = train_test_split(
        subject_targets.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=subject_targets.to_numpy(),
    )

    train_df = (
        df[df["id"].isin(train_subjects)]
        .copy()
        .sort_values(["id", "visit_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    test_df = (
        df[df["id"].isin(test_subjects)]
        .copy()
        .sort_values(["id", "visit_index"], kind="mergesort")
        .reset_index(drop=True)
    )

    X_train, feature_groups, _ = long_to_wide_frame(
        train_df,
        id_col="id",
        time_col="visit_index",
        longitudinal_columns=PBCSEQ_FEATURES,
    )
    X_test, _, _ = long_to_wide_frame(
        test_df,
        id_col="id",
        time_col="visit_index",
        longitudinal_columns=PBCSEQ_FEATURES,
    )
    X_test = X_test.reindex(columns=X_train.columns)
    y_train = train_df.groupby("id")["event"].first().loc[X_train.index]
    y_test = test_df.groupby("id")["event"].first().loc[X_test.index]

    def build_estimator(gamma: float) -> TpTDecisionTreeClassifier:
        return TpTDecisionTreeClassifier(
            gamma=gamma,
            criterion="entropy",
            max_depth=3,
            min_samples_leaf=3,
            min_samples_split=2,
            random_state=random_state,
            features_group=feature_groups,
        )

    best_gamma, estimator, gamma_table = select_gamma_by_tree_total_impurity(
        gamma_grid,
        build_estimator,
        X_train,
        y_train.to_numpy(),
    )
    gamma_profile = gamma_table.copy()
    gamma_table = deduplicate_gamma_table(gamma_table)

    truth = y_test.rename("observed")
    pred = pd.Series(
        estimator.predict(X_test),
        index=truth.index,
        name="predicted",
    )
    proba = pd.Series(
        estimator.predict_proba(X_test)[:, 1],
        index=truth.index,
        name="predicted_probability",
    )

    try:
        roc_auc = float(roc_auc_score(truth, proba))
    except ValueError:
        roc_auc = float("nan")

    tree_stats = summarize_tpt_tree(estimator.tree_)
    metrics = {
        "train_subjects": int(train_df["id"].nunique()),
        "test_subjects": int(test_df["id"].nunique()),
        "selected_gamma": best_gamma,
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "roc_auc": roc_auc,
        "tree_total_impurity": float(tree_stats["tree_total_impurity"]),
        "tree_mean_impurity": float(tree_stats["tree_mean_impurity"]),
        "tree_depth": int(estimator.tree_.max_depth),
        "n_nodes": int(estimator.tree_.node_count),
        "duration_nodes": int(tree_stats["duration_nodes"]),
        "duration_sample_weight": float(tree_stats["duration_sample_weight"]),
    }

    predictions = (
        pd.concat([truth, pred, proba], axis=1)
        .rename_axis("id")
        .reset_index()
        .sort_values("predicted_probability", ascending=False, kind="mergesort")
    )

    feature_names = list(X_train.columns)
    tree_text = export_text(
        estimator,
        feature_names=feature_names,
        decimals=2,
        max_depth=3,
    )

    params = hyperparameter_table(
        {
            "preprocessing": (
                f"subjects with at least {PBCSEQ_MIN_VISITS} visits; "
                f"keep up to {PBCSEQ_MAX_VISITS} visit-order waves"
            ),
            "criterion": "entropy",
            "candidate_gammas": f"{gamma_grid[0]:.3f} to {gamma_grid[-1]:.3f} by 0.001",
            "selected_gamma": best_gamma,
            "max_depth": 3,
            "min_samples_leaf": 3,
            "min_samples_split": 2,
        }
    )

    return ExampleResult(
        metrics=metrics,
        predictions=predictions,
        tree_text=tree_text,
        model=estimator,
        feature_names=feature_names,
        gamma_table=gamma_table,
        gamma_profile=gamma_profile,
        hyperparameters=params,
        long_preview=train_df[["id", "day", "visit_index", *PBCSEQ_FEATURES, "event"]]
        .head(8)
        .reset_index(drop=True),
        wide_preview=X_train.reset_index().head(5).reset_index(drop=True),
    )


def run_regression_example(
    *,
    gamma_grid: tuple[float, ...] = DEFAULT_REGRESSION_GAMMAS,
    test_size: float = 0.25,
    random_state: int = 42,
    max_depth: int = 4,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
) -> ExampleResult:
    """Fit and evaluate the Produc regression example."""

    df = load_produc_regression_data()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["state"]))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    X_train, feature_groups, _ = long_to_wide_frame(
        train_df,
        id_col="state",
        time_col="time_index",
        longitudinal_columns=PRODUC_FEATURES,
    )
    X_test, _, _ = long_to_wide_frame(
        test_df,
        id_col="state",
        time_col="time_index",
        longitudinal_columns=PRODUC_FEATURES,
    )
    y_train = train_df.groupby("state")["final_gsp"].first().loc[X_train.index].rename("final_gsp")
    y_test = test_df.groupby("state")["final_gsp"].first().loc[X_test.index].rename("observed")

    fitted_models: dict[float, TpTDecisionTreeRegressor] = {}
    gamma_records: list[dict[str, Any]] = []
    feature_names = list(X_train.columns)

    for gamma in gamma_grid:
        estimator = TpTDecisionTreeRegressor(
            gamma=float(gamma),
            criterion="squared_error",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state,
            features_group=feature_groups,
        )
        estimator.fit(X_train, y_train.to_numpy())
        fitted_models[float(gamma)] = estimator

        tree_stats = summarize_tpt_tree(estimator.tree_)
        split_features = np.asarray(estimator.tree_.feature)
        split_mask = split_features >= 0
        split_times = np.asarray(estimator.tree_.split_time_index)[split_mask].tolist()
        selected_features = [feature_names[index] for index in split_features[split_mask]]

        gamma_records.append(
            {
                "gamma": float(gamma),
                "tree_total_impurity": tree_stats["tree_total_impurity"],
                "tree_mean_impurity": tree_stats["tree_mean_impurity"],
                "root_feature": selected_features[0] if selected_features else "leaf",
                "root_time": split_times[0] if split_times else np.nan,
                "min_split_time": min(split_times) if split_times else np.nan,
                "max_split_time": max(split_times) if split_times else np.nan,
                "tree_depth": tree_stats["tree_depth"],
                "n_nodes": tree_stats["n_nodes"],
            }
        )

    gamma_table = pd.DataFrame(gamma_records).sort_values("gamma", kind="mergesort").reset_index(drop=True)
    best_row = (
        gamma_table.sort_values(
            ["tree_total_impurity", "tree_mean_impurity", "gamma"],
            kind="mergesort",
        )
        .iloc[0]
    )
    best_gamma = float(best_row["gamma"])
    estimator = fitted_models[best_gamma]

    pred = pd.Series(
        estimator.predict(X_test),
        index=y_test.index,
        name="predicted",
    )

    tree_stats = summarize_tpt_tree(estimator.tree_)
    metrics = {
        "train_subjects": int(len(y_train)),
        "test_subjects": int(len(y_test)),
        "selected_gamma": best_gamma,
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "tree_total_impurity": float(tree_stats["tree_total_impurity"]),
        "tree_mean_impurity": float(tree_stats["tree_mean_impurity"]),
        "tree_depth": int(estimator.tree_.max_depth),
        "n_nodes": int(estimator.tree_.node_count),
    }

    predictions = (
        pd.concat([y_test, pred], axis=1)
        .rename_axis("state")
        .reset_index()
        .assign(abs_error=lambda frame: (frame["observed"] - frame["predicted"]).abs())
        .sort_values("abs_error", ascending=False, kind="mergesort")
    )

    tree_text = export_text(
        estimator,
        feature_names=feature_names,
        decimals=2,
        max_depth=3,
    )

    preview_columns = [
        "subject_id",
        "pc_t0",
        "pc_t5",
        "pc_t15",
        "emp_t0",
        "emp_t5",
        "unemp_t5",
        "final_gsp",
    ]
    wide_preview = (
        X_train.assign(subject_id=X_train.index, final_gsp=y_train.to_numpy())
        .loc[:, preview_columns]
        .head(5)
        .reset_index(drop=True)
    )

    params = hyperparameter_table(
        {
            "input_format": "wide after LongitudinalDataset.to_wide",
            "criterion": "squared_error (MSE)",
            "candidate_gammas": f"{gamma_grid[0]:.3f} to {gamma_grid[-1]:.3f} by 0.001",
            "selected_gamma": best_gamma,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "longitudinal_predictors": ", ".join(PRODUC_FEATURES),
            "target": "final_gsp at the held-out final year",
            "selection_rule": "minimize training tree_total_impurity on a 0.001 grid",
        }
    )

    return ExampleResult(
        metrics=metrics,
        predictions=predictions,
        tree_text=tree_text,
        model=estimator,
        feature_names=feature_names,
        gamma_table=gamma_table,
        gamma_profile=gamma_table.copy(),
        hyperparameters=params,
        long_preview=(
            train_df[["state", "year", "time_index", "duration", *PRODUC_FEATURES, "final_gsp"]]
            .head(5)
            .reset_index(drop=True)
        ),
        wide_preview=wide_preview,
        stage_predictions=stagewise_history_predictions(estimator, X_test, y_test),
    )
