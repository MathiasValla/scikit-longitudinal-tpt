"""One landmark prediction case study, using public pbcseq visit histories."""

from dataclasses import dataclass
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="pkg_resources is deprecated as an API.*", module="stopit"
    )
    from scikit_longitudinal.data_preparation import LongitudinalDataset
    from scikit_longitudinal.estimators.trees import TpTDecisionTreeClassifier


RAW = Path(__file__).resolve().parents[1] / "data/raw/pbcseq.csv"
BIOMARKERS = ["bili", "albumin", "protime", "ast"]
LANDMARK_DAY, HORIZON_DAY = 1826, 3652
PARAMS = dict(criterion="entropy", max_depth=3, min_samples_leaf=20, random_state=42)
GAMMA = 0.1
SEEDS = tuple(range(10))
COLORS = {"TpT": "#006b73", "Last-visit CART": "#9b472e", "CART + visit metadata": "#465676"}
INK = "#23333b"


@dataclass
class Cohort:
    visits: pd.DataFrame
    wide: pd.DataFrame
    groups: list
    last: pd.DataFrame
    y: np.ndarray
    flow: dict


def load_cohort(landmark=LANDMARK_DAY, horizon=HORIZON_DAY):
    raw = pd.read_csv(RAW).sort_values(["id", "day"])
    follow_up = raw.drop_duplicates("id").set_index("id")[["futime", "status"]]
    post_landmark = follow_up.futime.gt(landmark)
    label_known = follow_up.status.ne(0) | follow_up.futime.ge(horizon)
    eligible = follow_up.loc[post_landmark & label_known]
    visits = raw.loc[raw.day.le(landmark) & raw.id.isin(eligible.index)].copy()
    visits["wave"] = visits.groupby("id").cumcount()
    if visits[BIOMARKERS].isna().any().any():
        raise ValueError("The case study permits absent visits, not missing biomarkers.")

    dataset = LongitudinalDataset(file_path=None, data_frame=visits[["id", "wave", *BIOMARKERS]])
    wide = dataset.to_wide(
        id_col="id", time_col="wave", longitudinal_columns=BIOMARKERS,
        wave_format="{feature}_t{wave}",
    )
    groups = dataset.feature_groups()
    last = visits.drop_duplicates("id", keep="last").set_index("id").loc[wide.index]
    eligible = eligible.loc[wide.index]
    y = (eligible.status.ne(0) & eligible.futime.le(horizon)).to_numpy(dtype=int)
    flow = {
        "original": len(follow_up),
        "not_after_landmark": int((~post_landmark).sum()),
        "unknown_endpoint": int((post_landmark & ~label_known).sum()),
        "eligible": len(eligible),
    }
    assert flow["original"] - flow["not_after_landmark"] - flow["unknown_endpoint"] == len(y)
    return Cohort(visits, wide, groups, last, y, flow)


def fit_tpt(cohort, rows, gamma=GAMMA):
    model = TpTDecisionTreeClassifier(features_group=cohort.groups, gamma=gamma, **PARAMS)
    model.fit(cohort.wide.iloc[rows], cohort.y[rows])
    validate_tpt_fit(model, cohort.wide.iloc[rows], cohort.y[rows])
    return model


def validate_tpt_fit(model, X, y):
    """Verify stored terminal sizes and probabilities against prediction routing."""
    tree = model.tree_
    terminal = model.apply(X)
    ordinary = np.flatnonzero(tree.children_left == -1)
    duration = np.flatnonzero(tree.n_duration_samples)
    expected = set(ordinary) | set(-duration - 1)
    assert set(terminal) == expected
    np.testing.assert_array_equal(tree.n_duration_samples, tree.weighted_n_duration)
    probabilities = model.predict_proba(X)
    for node in expected:
        labels = y[terminal == node]
        size = tree.n_duration_samples[-node - 1] if node < 0 else tree.n_node_samples[node]
        assert len(labels) == size
        expected_proba = np.bincount(labels, minlength=2) / size
        np.testing.assert_allclose(
            probabilities[terminal == node], np.tile(expected_proba, (size, 1))
        )


def predict_oof(cohort, seed, gamma=GAMMA):
    X, y = cohort.wide, cohort.y
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = pd.DataFrame(np.nan, index=X.index, columns=list(COLORS))
    uses_duration = np.zeros(len(y), dtype=bool)
    fold = np.zeros(len(y), dtype=int)
    for fold_id, (train, test) in enumerate(splits.split(X, y), start=1):
        model = fit_tpt(cohort, train, gamma)
        scores.loc[X.index[test], "TpT"] = model.predict_proba(X.iloc[test])[:, 1]
        uses_duration[test] = model.apply(X.iloc[test]) < 0
        fold[test] = fold_id
        for name, columns in [
            ("Last-visit CART", BIOMARKERS),
            ("CART + visit metadata", [*BIOMARKERS, "wave", "day"]),
        ]:
            last = cohort.last[columns]
            cart = DecisionTreeClassifier(**PARAMS).fit(last.iloc[train], y[train])
            scores.loc[X.index[test], name] = cart.predict_proba(last.iloc[test])[:, 1]
    assert scores.notna().all().all() and (fold > 0).all()
    return scores.assign(outcome=y, fold=fold, duration=uses_duration)


def run_cv(cohort, seeds=SEEDS, gamma=GAMMA):
    predictions = {seed: predict_oof(cohort, seed, gamma) for seed in seeds}
    aucs = pd.DataFrame(
        {seed: {name: roc_auc_score(pred.outcome, pred[name]) for name in COLORS}
         for seed, pred in predictions.items()}
    ).T
    return predictions, aucs


def auc_summary(aucs):
    return (
        pd.DataFrame({
            "Mean AUC": aucs.mean(),
            "Min. AUC": aucs.min(),
            "Max. AUC": aucs.max(),
        })
        .round(3)
        .rename(index={"CART + visit metadata": "CART with timing"})
        .rename_axis("Model")
        .reset_index()
    )


def tree_summary(model):
    tree = model.tree_
    return {
        "split_waves": sorted(set(tree.split_time_index[tree.feature >= 0].tolist())),
        "ordinary_leaves": model.get_n_leaves(),
        "duration_leaves": int(np.count_nonzero(tree.n_duration_samples)),
        "duration_patients": int(tree.n_duration_samples.sum()),
    }


def plot_roc(predictions, aucs):
    """Show the first repeat's ROC and the paired contrast across repeats."""
    style = {
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#839099",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
    }
    with plt.rc_context(style):
        fig, (ax, paired) = plt.subplots(
            ncols=2, figsize=(7.2, 3.55), width_ratios=[1.22, 1], layout="constrained"
        )
        ax.set_title("(a) ROC, first repeat", loc="left", fontweight="semibold")
        ax.plot([0, 1], [0, 1], linestyle=":", color="#929ca3", linewidth=1.2, zorder=1)
        for (name, color), linestyle in zip(COLORS.items(), ["-", "--", "-."]):
            fpr, tpr, _ = roc_curve(predictions.outcome, predictions[name])
            auc = roc_auc_score(predictions.outcome, predictions[name])
            focal = name == "TpT"
            label = "CART with timing" if name == "CART + visit metadata" else name
            ax.plot(fpr, tpr, color=color, ls=linestyle, lw=2.3 if focal else 1.65,
                    zorder=4 if focal else 3, label=f"{label} ({auc:.3f})")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               xlim=(0, 1), ylim=(0, 1.02))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e1e7eb", linewidth=0.65)
        ax.tick_params(length=3)
        ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=2.2)

        difference = aucs["TpT"] - aucs["Last-visit CART"]
        repeats = np.arange(1, len(difference) + 1)
        paired.set_title("(b) Paired AUC difference", loc="left", fontweight="semibold")
        paired.axvline(0, color="#929ca3", linewidth=1)
        paired.axvline(difference.mean(), ls="--", lw=1, color=COLORS["TpT"], alpha=0.75)
        paired.scatter(difference, repeats, s=25, color=COLORS["TpT"], zorder=3)
        paired.set(xlabel="AUC difference", ylabel="Repeat",
                   xlim=(min(0, difference.min()) - 0.012, max(0, difference.max()) + 0.012),
                   ylim=(len(repeats) + 0.6, 0.4), yticks=repeats)
        paired.xaxis.set_major_locator(MaxNLocator(4))
        paired.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        paired.set_axisbelow(True)
        paired.grid(axis="y", color="#e1e7eb", linewidth=0.65)
        paired.tick_params(length=3)
    return fig


def plot_tpt(model, columns):
    """Relabel the exported tree for the article, keeping its nodes and edges."""
    tree = model.tree_
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.96)
    artists = plot_tree(
        model, feature_names=list(columns), node_ids=True, impurity=False,
        rounded=True, fontsize=22, tpt_time_scale=0.18, ax=ax,
    )
    node_at = {
        artist.get_position(): int(artist.get_text().splitlines()[0].split("#", 1)[1])
        for artist in artists if artist.get_text().startswith("node #")
    }
    duration_at = {
        artist.get_position(): artist
        for artist in artists if artist.get_text().startswith("Duration\n")
    }
    event_fill = "#d0e8f1"
    non_event_fill = "#f3e1bf"
    duration_fill = "#f1f3f4"
    edge = "#536873"
    root_y = next(y for (_, y), node in node_at.items() if node == 0)

    def expanded_x(x):
        return 1.3 * x + 0.1

    def event_label(size, probability):
        return f"{probability:.0%} ({round(size * probability)}/{size})"

    for artist in ax.texts:
        artist.set_color(INK)
    for artist in artists:
        box = artist.get_bbox_patch()
        if box is not None:
            box.set_edgecolor(edge)
            box.set_linewidth(0.9)
        if artist.arrow_patch is not None:
            artist.arrow_patch.set_color(edge)
        position = artist.get_position()
        if position in duration_at:
            continue
        x, y = position
        parent_x, parent_y = artist.xy
        artist.set_position((expanded_x(x), y))
        artist.xy = (expanded_x(parent_x), parent_y)
        if box is None:
            artist.set_visible(False)
            continue
        node = node_at[position]
        if position != (parent_x, parent_y):
            parent = node_at[(parent_x, parent_y)]
            is_left = node == tree.children_left[parent]
            ax.text(
                expanded_x((x + parent_x) / 2) + (-0.018 if is_left else 0.018),
                (y + parent_y) / 2 + 0.018, "Yes" if is_left else "No",
                ha="right" if is_left else "left", va="center",
                fontsize=9.5, color=INK, transform=ax.transAxes,
            )
        size = tree.n_node_samples[node]
        feature = tree.feature[node]
        if feature < 0:
            probability = tree.value[node, 0, 1]
            artist.set_text(event_label(size, probability))
            box.set_facecolor(event_fill if probability >= 0.5 else non_event_fill)
        else:
            wave = tree.split_time_index[node]
            biomarker = columns[feature].rsplit("_t", 1)[0]
            label = {"bili": "bilirubin", "protime": "prothrombin time"}.get(biomarker, biomarker)
            threshold = f"{tree.threshold[node]:.4g}"
            artist.set_text(f"Visit {wave}: {label} " + rf"$\leq$ {threshold}" + f"\n{size} patients")
            box.set_facecolor("#ffffff")

    # Move each duration box and its dashed edge together.
    for link in ax.lines:
        link.set_color(edge)
        if link.get_linestyle() == "--":
            (parent_x, parent_y), duration_position = map(tuple, link.get_xydata())
            node = node_at[(parent_x, parent_y)]
            artist = duration_at[duration_position]
            size = tree.n_duration_samples[node]
            probability = tree.duration_value[node, 0, 1]
            wave = tree.split_time_index[node]
            artist.set_text(f"Visit {wave} unavailable\n{event_label(size, probability)}")
            artist.get_bbox_patch().set_facecolor(duration_fill)
            x = expanded_x(parent_x) + 0.33
            artist.xy = (x, parent_y)
            artist.set_position(artist.xy)
            link.set_xdata([expanded_x(parent_x), x])
        elif np.all(link.get_xdata() == 0):
            link.set_ydata([0, root_y + 0.10])
    ax.text(-0.025, root_y + 0.14, "Visit index\nreached",
            transform=ax.transAxes, fontsize=9, color=INK)
    ax.legend(
        handles=[
            Patch(facecolor=event_fill, label="Event majority"),
            Patch(facecolor=non_event_fill, label="No-event majority"),
            Patch(facecolor=duration_fill, edgecolor=edge, ls="--", label="Duration leaf"),
        ],
        loc="lower center", bbox_to_anchor=(0.53, -0.115), ncol=3,
        frameon=False, labelcolor=INK, fontsize=9.5,
    )
    return fig
