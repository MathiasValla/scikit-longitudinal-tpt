"""Reusable helpers for the TpT Computo paper examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import _tree, export_text

from scikit_longitudinal.estimators.trees import (
    TpTDecisionTreeClassifier,
    TpTDecisionTreeRegressor,
)
from scikit_longitudinal.estimators.trees.TpT._preprocessing import long_to_wide


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

PBCSEQ_PATH = DATA_DIR / "pbcseq_classification_long.csv"
DIETOX_PATH = DATA_DIR / "dietox_regression_long.csv"

PBCSEQ_FEATURES = ["bili", "albumin", "protime", "ast"]
DIETOX_FEATURES = ["Weight"]


def gamma_grid(max_gamma: float, step: float = 0.005) -> tuple[float, ...]:
    """Build a regular gamma grid including the upper bound."""

    n_steps = int(round(max_gamma / step))
    return tuple(round(step * idx, 6) for idx in range(n_steps + 1))


DEFAULT_CLASSIFICATION_GAMMAS = gamma_grid(0.4, step=0.005)
DEFAULT_REGRESSION_GAMMAS = gamma_grid(0.2, step=0.005)


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
    long_preview: pd.DataFrame | None = None
    wide_preview: pd.DataFrame | None = None
    stage_predictions: pd.DataFrame | None = None


def load_pbcseq_classification_data() -> pd.DataFrame:
    """Load the processed pbcseq classification dataset."""

    return pd.read_csv(PBCSEQ_PATH)


def load_dietox_regression_data() -> pd.DataFrame:
    """Load the processed dietox regression dataset."""

    return pd.read_csv(DIETOX_PATH)


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

    df = load_pbcseq_classification_data()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["id"]))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    feature_columns = ["id", "day", "futime", *PBCSEQ_FEATURES]

    def build_estimator(gamma: float) -> TpTDecisionTreeClassifier:
        return TpTDecisionTreeClassifier(
            gamma=gamma,
            max_depth=3,
            min_samples_leaf=10,
            random_state=random_state,
            assume_long_format=True,
            id_col="id",
            time_col="day",
            duration_col="futime",
            time_step=365.0,
            long_feature_columns=PBCSEQ_FEATURES,
        )

    best_gamma, estimator, gamma_table = select_gamma_by_tree_total_impurity(
        gamma_grid,
        build_estimator,
        train_df[feature_columns],
        train_df["event"],
    )
    gamma_table = deduplicate_gamma_table(gamma_table)

    truth = test_df.groupby("id")["event"].first().rename("observed")
    pred = pd.Series(
        estimator.predict(test_df[feature_columns]),
        index=truth.index,
        name="predicted",
    )
    proba = pd.Series(
        estimator.predict_proba(test_df[feature_columns])[:, 1],
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
    }

    predictions = (
        pd.concat([truth, pred, proba], axis=1)
        .rename_axis("id")
        .reset_index()
        .sort_values("predicted_probability", ascending=False, kind="mergesort")
    )

    feature_names = list(estimator._wide_feature_names_)
    tree_text = export_text(
        estimator,
        feature_names=feature_names,
        decimals=2,
        max_depth=3,
    )

    params = hyperparameter_table(
        {
            "input_format": "long",
            "criterion": "gini",
            "candidate_gammas": f"{gamma_grid[0]:.3f} to {gamma_grid[-1]:.3f} by 0.005",
            "selected_gamma": best_gamma,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "time_step_days": 365.0,
            "selection_rule": "minimize training tree_total_impurity on a 0.005 grid",
        }
    )

    return ExampleResult(
        metrics=metrics,
        predictions=predictions,
        tree_text=tree_text,
        model=estimator,
        feature_names=feature_names,
        gamma_table=gamma_table,
        hyperparameters=params,
        long_preview=train_df[feature_columns + ["event"]].head(8).reset_index(drop=True),
        wide_preview=None,
    )


def run_regression_example(
    *,
    gamma_grid: tuple[float, ...] = DEFAULT_REGRESSION_GAMMAS,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ExampleResult:
    """Fit and evaluate the dietox regression example."""

    df = load_dietox_regression_data()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["Pig"]))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    long_columns = ["Pig", "Time", "duration", *DIETOX_FEATURES]

    wide_train = long_to_wide(
        train_df[long_columns],
        id_col="Pig",
        time_col="Time",
        duration_col="duration",
        time_step=1.0,
        feature_columns=DIETOX_FEATURES,
        target=train_df["final_weight"],
    )
    wide_test = long_to_wide(
        test_df[long_columns],
        id_col="Pig",
        time_col="Time",
        duration_col="duration",
        time_step=1.0,
        feature_columns=DIETOX_FEATURES,
        target=test_df["final_weight"],
    )

    X_train = pd.DataFrame(wide_train.X, columns=wide_train.feature_names)
    X_test = pd.DataFrame(wide_test.X, columns=wide_test.feature_names)
    y_train = pd.Series(wide_train.y, index=wide_train.subject_ids, name="final_weight")
    y_test = pd.Series(wide_test.y, index=wide_test.subject_ids, name="observed")

    def build_estimator(gamma: float) -> TpTDecisionTreeRegressor:
        return TpTDecisionTreeRegressor(
            gamma=gamma,
            criterion="squared_error",
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state,
            features_group=wide_train.feature_groups,
        )

    best_gamma, estimator, gamma_table = select_gamma_by_tree_total_impurity(
        gamma_grid,
        build_estimator,
        X_train,
        y_train.to_numpy(),
    )
    gamma_table = deduplicate_gamma_table(gamma_table)

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
        .rename_axis("Pig")
        .reset_index()
        .assign(abs_error=lambda frame: (frame["observed"] - frame["predicted"]).abs())
        .sort_values("abs_error", ascending=False, kind="mergesort")
    )

    feature_names = list(wide_train.feature_names)
    tree_text = export_text(
        estimator,
        feature_names=feature_names,
        decimals=2,
        max_depth=3,
    )

    wide_preview = (
        X_train.assign(subject_id=wide_train.subject_ids, final_weight=y_train.to_numpy())
        .loc[:, ["subject_id", "Weight_t0", "Weight_t1", "Weight_t2", "Weight_t3", "final_weight"]]
        .head(8)
        .reset_index(drop=True)
    )

    params = hyperparameter_table(
        {
            "input_format": "wide (after explicit long_to_wide conversion)",
            "criterion": "squared_error (MSE)",
            "candidate_gammas": f"{gamma_grid[0]:.3f} to {gamma_grid[-1]:.3f} by 0.005",
            "selected_gamma": best_gamma,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "time_step": 1.0,
            "selection_rule": "minimize training tree_total_impurity on a 0.005 grid",
        }
    )

    return ExampleResult(
        metrics=metrics,
        predictions=predictions,
        tree_text=tree_text,
        model=estimator,
        feature_names=feature_names,
        gamma_table=gamma_table,
        hyperparameters=params,
        long_preview=train_df[long_columns + ["final_weight"]].head(8).reset_index(drop=True),
        wide_preview=wide_preview,
        stage_predictions=stagewise_history_predictions(estimator, X_test, y_test),
    )
