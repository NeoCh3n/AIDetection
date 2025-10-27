"""
t-SNE & Model Diagnostic Visualization Utilities
================================================

Provides helper routines to visualise model behaviour and evaluation metrics:

* t-SNE projection of high-dimensional feature vectors.
* ROC, cumulative gains, lift, and calibration curves.
* Confusion matrix heatmap, classification report (text file), and
  feature-importance bar chart.

All plots are rendered with matplotlib's Agg backend so they work in headless
environments (e.g., CI, remote servers). Outputs are saved as PNG files to the
specified directory (typically the ``model`` folder used by the training
pipeline).
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)

# Ensure plots can be rendered in environments without a display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

logger = logging.getLogger(__name__)


class TSNEConfig(object):
    """Lightweight configuration container for t-SNE settings."""

    def __init__(
        self,
        perplexity=30.0,
        learning_rate=200.0,
        n_iter=1000,
        metric="euclidean",
        random_state=42,
        n_components=3,
    ):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.random_state = random_state
        self.n_components = n_components


def generate_tsne_plot(
    features: np.ndarray,
    labels: Iterable[int],
    output_path: Optional[str] = None,
    config: Optional["TSNEConfig"] = None,
) -> Dict[str, Path]:
    """
    Render t-SNE scatter plots (2D and 3D) for the provided feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (samples x features) used during training.
    labels : Iterable[int]
        Binary labels aligned with the feature rows (0 = normal, 1 = attack).
    output_path : Optional[str]
        Base path for the resulting PNG(s). Defaults to ``tsne_scatter.png`` when
        not specified. The 2D image is written to ``<stem>_2d.png`` and the 3D image
        to ``<stem>.png`` (to preserve backwards compatibility).
    config : Optional[TSNEConfig]
        Optional custom configuration for t-SNE hyperparameters.

    Returns
    -------
    Dict[str, Path]
        Mapping of dimension ("2d"/"3d") to generated PNG paths.
    """

    if features is None or len(features) == 0:
        raise ValueError("Cannot generate t-SNE plot: feature matrix is empty")

    features = np.asarray(features)
    labels_array = np.asarray(list(labels))
    if features.shape[0] != labels_array.shape[0]:
        raise ValueError(
            "Features and labels mismatch: %s samples vs %s labels"
            % (features.shape[0], labels_array.shape[0])
        )

    cfg = config or TSNEConfig()

    base_output = Path(output_path) if output_path else Path("tsne_scatter.png")
    base_output.parent.mkdir(parents=True, exist_ok=True)
    output_2d = base_output.with_name(f"{base_output.stem}_2d{base_output.suffix}")
    output_3d = base_output

    artifacts: Dict[str, Path] = {}

    def _fit_tsne(n_components: int) -> np.ndarray:
        logger.info(
            "Running t-SNE with perplexity=%s, learning_rate=%s, n_iter=%s, n_components=%s",
            cfg.perplexity,
            cfg.learning_rate,
            cfg.n_iter,
            n_components,
        )
        tsne = TSNE(
            n_components=n_components,
            perplexity=cfg.perplexity,
            learning_rate=cfg.learning_rate,
            n_iter=cfg.n_iter,
            metric=cfg.metric,
            random_state=cfg.random_state,
            init="pca",
            verbose=0,
        )
        emb = tsne.fit_transform(features)
        logger.debug("t-SNE embedding shape (n=%s): %s", n_components, emb.shape)
        return emb

    # 2D projection ---------------------------------------------------------
    try:
        emb_2d = _fit_tsne(2)
        plt.figure(figsize=(8, 6))
        is_attack = labels_array == 1
        plt.scatter(
            emb_2d[~is_attack, 0],
            emb_2d[~is_attack, 1],
            c="royalblue",
            alpha=0.6,
            label="Normal (0)",
            edgecolors="none",
        )
        plt.scatter(
            emb_2d[is_attack, 0],
            emb_2d[is_attack, 1],
            c="crimson",
            alpha=0.7,
            label="Attack (1)",
            edgecolors="none",
        )
        plt.title("t-SNE Projection of Training Windows (2D)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_2d, dpi=300, bbox_inches="tight")
        plt.close()
        artifacts["2d"] = output_2d
        logger.info("t-SNE 2D scatter plot saved to %s", output_2d)
    except Exception as exc:
        logger.warning("Failed to generate 2D t-SNE plot: %s", exc)

    # 3D projection ---------------------------------------------------------
    try:
        n_components_3d = max(3, int(getattr(cfg, "n_components", 3)))
        emb_3d = _fit_tsne(n_components_3d)
        if emb_3d.shape[1] < 3:
            raise ValueError(
                "t-SNE returned %s dimensions, cannot render 3D plot"
                % emb_3d.shape[1]
            )
        is_attack = labels_array == 1
        class_def: List[Tuple[np.ndarray, str, str]] = [
            (emb_3d[~is_attack], "royalblue", "Normal (0)"),
            (emb_3d[is_attack], "crimson", "Attack (1)"),
        ]

        fig = plt.figure(figsize=(12, 10))
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        for coords, color, label in class_def:
            if coords.size == 0:
                continue
            ax3d.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=color,
                alpha=0.7,
                label=label,
                edgecolors="none",
                depthshade=True,
            )
        ax3d.set_title("t-SNE Projection (3D View)")
        ax3d.set_xlabel("Component 1")
        ax3d.set_ylabel("Component 2")
        ax3d.set_zlabel("Component 3")
        ax3d.legend(loc="best")
        ax3d.grid(True)

        def _plot_projection(ax, dims: Tuple[int, int], title: str):
            x_idx, y_idx = dims
            for coords, color, label in class_def:
                if coords.size == 0:
                    continue
                ax.scatter(
                    coords[:, x_idx],
                    coords[:, y_idx],
                    c=color,
                    alpha=0.7,
                    label=label,
                    edgecolors="none",
                )
            ax.set_title(title)
            ax.set_xlabel(f"Component {x_idx + 1}")
            ax.set_ylabel(f"Component {y_idx + 1}")
            ax.grid(True, linestyle="--", alpha=0.3)

        ax_top = fig.add_subplot(2, 2, 2)
        _plot_projection(ax_top, (0, 1), "Top View (Comp1 vs Comp2)")

        ax_front = fig.add_subplot(2, 2, 3)
        _plot_projection(ax_front, (0, 2), "Front View (Comp1 vs Comp3)")

        ax_side = fig.add_subplot(2, 2, 4)
        _plot_projection(ax_side, (1, 2), "Side View (Comp2 vs Comp3)")

        # Only keep a single legend (top-right subplot) to reduce clutter
        handles, labels = ax_top.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles))
            for axis in (ax_top, ax_front, ax_side):
                legend_obj = axis.get_legend()
                if legend_obj is not None:
                    legend_obj.remove()

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(output_3d, dpi=300, bbox_inches="tight")
        plt.close(fig)
        artifacts["3d"] = output_3d
        logger.info("t-SNE 3D scatter plot saved to %s", output_3d)
    except Exception as exc:
        logger.warning("Failed to generate 3D t-SNE plot: %s", exc)

    return artifacts


def generate_evaluation_artifacts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[Sequence[float]],
    output_dir: Path,
    classification_report_text: Optional[str] = None,
    feature_importances: Optional[Sequence[float]] = None,
    feature_names: Optional[Sequence[str]] = None,
    top_k_features: int = 20,
) -> Dict[str, Path]:
    """
    Generate a suite of diagnostic plots and reports after model training.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels.
    y_pred : Sequence[int]
        Model-predicted binary labels.
    y_proba : Optional[Sequence[float]]
        Positive-class probabilities. Required for ROC, gains, lift, and
        calibration curves. Plots depending on probabilities are skipped when
        this is ``None``.
    output_dir : Path
        Directory where PNG/text artifacts are written.
    classification_report_text : Optional[str]
        Full classification report (precision/recall/F1) for persistence.
    feature_importances : Optional[Sequence[float]]
        Feature importance values (e.g., from RandomForest.feature_importances_).
    feature_names : Optional[Sequence[str]]
        Names matching ``feature_importances`` indices. Falls back to generic
        names if not provided.
    top_k_features : int
        Maximum number of features to show in the importance bar chart.

    Returns
    -------
    Dict[str, Path]
        Mapping of artifact names to generated file paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    y_proba_arr = np.asarray(list(y_proba)) if y_proba is not None else None

    artifacts: Dict[str, Path] = {}

    # ROC Curve -------------------------------------------------------------
    if y_proba_arr is not None and y_proba_arr.size == y_true_arr.size:
        try:
            roc_path = output_dir / "roc_curve.png"
            _plot_roc_curve(y_true_arr, y_proba_arr, roc_path)
            artifacts["roc_curve"] = roc_path
        except Exception as exc:
            logger.warning("Failed to generate ROC curve: %s", exc)

        # Cumulative Gains & Lift Charts share the same computed arrays
        try:
            gains_path = output_dir / "cumulative_gains_curve.png"
            lift_path = output_dir / "lift_chart.png"
            perc_samples, perc_positives, lift_values = _compute_cumulative_gains(
                y_true_arr, y_proba_arr
            )
            _plot_cumulative_gains(perc_samples, perc_positives, gains_path)
            _plot_lift_chart(perc_samples, lift_values, lift_path)
            artifacts["cumulative_gains"] = gains_path
            artifacts["lift_chart"] = lift_path
        except Exception as exc:
            logger.warning("Failed to generate gains/lift charts: %s", exc)

        try:
            calibration_path = output_dir / "calibration_curve.png"
            _plot_calibration_curve(y_true_arr, y_proba_arr, calibration_path)
            artifacts["calibration_curve"] = calibration_path
        except Exception as exc:
            logger.warning("Failed to generate calibration curve: %s", exc)

    else:
        logger.warning(
            "Probability scores unavailable; skipping ROC, gains, lift, and calibration plots"
        )

    # Confusion Matrix ------------------------------------------------------
    try:
        cm_path = output_dir / "confusion_matrix.png"
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        _plot_confusion_matrix(cm, cm_path)
        artifacts["confusion_matrix"] = cm_path
    except Exception as exc:
        logger.warning("Failed to generate confusion matrix: %s", exc)

    # Classification Report -------------------------------------------------
    if classification_report_text:
        try:
            report_path = output_dir / "classification_report.txt"
            report_path.write_text(classification_report_text, encoding="utf-8")
            artifacts["classification_report"] = report_path
        except Exception as exc:
            logger.warning("Failed to persist classification report: %s", exc)

    # Feature Importance ----------------------------------------------------
    if feature_importances is not None:
        try:
            importance_path = output_dir / "feature_importance.png"
            _plot_feature_importance(
                np.asarray(list(feature_importances)),
                feature_names,
                importance_path,
                top_k_features,
            )
            artifacts["feature_importance"] = importance_path
        except Exception as exc:
            logger.warning("Failed to generate feature-importance chart: %s", exc)

    return artifacts


def _plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (AUC = %.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _compute_cumulative_gains(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]

    total_samples = y_true_sorted.size
    if total_samples == 0:
        raise ValueError("No samples available for cumulative gains computation")

    positives_total = np.sum(y_true_sorted)
    if positives_total <= 0:
        positives_total = 1  # avoid division by zero, curve will stay near zero

    cum_positives = np.cumsum(y_true_sorted)
    perc_samples = np.arange(1, total_samples + 1, dtype=float) / float(total_samples)
    perc_positives = cum_positives / float(positives_total)

    expected_positive_rate = np.arange(1, total_samples + 1, dtype=float) / float(
        total_samples
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(
            expected_positive_rate > 0,
            perc_positives / expected_positive_rate,
            0.0,
        )

    return perc_samples, perc_positives, lift


def _plot_cumulative_gains(
    perc_samples: np.ndarray, perc_positives: np.ndarray, output_path: Path
) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(
        perc_samples * 100,
        perc_positives * 100,
        color="green",
        lw=2,
        label="Model",
    )
    plt.plot(
        [0, 100],
        [0, 100],
        color="gray",
        lw=1,
        linestyle="--",
        label="Baseline",
    )
    plt.xlim([0, 100])
    plt.ylim([0, 105])
    plt.xlabel("Cumulative Percentage of Samples")
    plt.ylabel("Cumulative Percentage of Positives")
    plt.title("Cumulative Gains Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_lift_chart(
    perc_samples: np.ndarray, lift_values: np.ndarray, output_path: Path
) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(
        perc_samples * 100,
        lift_values,
        color="purple",
        lw=2,
        label="Model Lift",
    )
    plt.axhline(1.0, color="gray", linestyle="--", label="Baseline Lift = 1")
    plt.xlim([0, 100])
    plt.ylim(bottom=0)
    plt.xlabel("Cumulative Percentage of Samples")
    plt.ylabel("Lift")
    plt.title("Lift Chart")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_calibration_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_path: Path
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, ["Normal", "Attack"])
    plt.yticks(tick_marks, ["Normal", "Attack"])

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_feature_importance(
    feature_importances: np.ndarray,
    feature_names: Optional[Sequence[str]],
    output_path: Path,
    top_k: int,
) -> None:
    feature_importances = np.asarray(feature_importances, dtype=float).ravel()
    if feature_importances.ndim != 1:
        raise ValueError("feature_importances must be a 1-D sequence")

    if feature_importances.size == 0:
        raise ValueError("feature_importances is empty")

    top_k = max(1, int(top_k))

    # Sort by importance descending
    order = np.argsort(feature_importances)[::-1]
    ordered_pairs = [(idx, feature_importances[idx]) for idx in order]

    # Prefer strictly positive importances for visibility
    positive_pairs = [(idx, val) for idx, val in ordered_pairs if np.isfinite(val) and val > 0]
    selected = positive_pairs[:top_k] if positive_pairs else ordered_pairs[:top_k]

    if not selected:
        raise ValueError("Unable to select feature importances for plotting")

    sel_indices, sel_values = zip(*selected)
    sel_indices_arr = np.array(sel_indices, dtype=int)
    sel_values_arr = np.array(sel_values, dtype=float)

    if feature_names is not None and len(feature_names) >= feature_importances.size:
        names = [str(feature_names[i]) for i in sel_indices_arr]
    else:
        names = ["Feature %d" % i for i in sel_indices_arr]

    # Preserve descending order top -> bottom in bar chart
    raw_values = sel_values_arr[::-1]
    plot_names = names[::-1]
    y_pos = np.arange(len(raw_values))

    # Normalise purely for visual clarity when magnitudes are tiny
    max_abs = np.max(np.abs(raw_values))
    if max_abs > 0:
        plot_values = raw_values
        display_values = raw_values
    else:
        # All importances zero (or effectively zero). Use equal bars for visibility
        plot_values = np.ones_like(raw_values)
        display_values = raw_values

    plt.figure(figsize=(8, max(4, len(plot_values) * 0.4)))
    bars = plt.barh(y_pos, plot_values, align="center", color="steelblue")
    plt.yticks(y_pos, plot_names)
    plt.xlabel("Importance")
    plt.title("Top %d Feature Importances" % len(plot_values))
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    # Annotate each bar with the raw (un-normalised) importance
    for bar, raw in zip(bars, display_values):
        width = bar.get_width()
        text_x = width if width > 0 else bar.get_x() + 0.02
        plt.text(
            text_x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{raw:.4e}",
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
