"""
t-SNE Visualization Utilities
=============================

Generates scatter plots that project high-dimensional feature vectors onto a
2D plane using t-Distributed Stochastic Neighbour Embedding (t-SNE). Designed
to run in headless environments and integrate with the supervised training
pipeline.
"""

import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
import numpy as np
from sklearn.manifold import TSNE

# Ensure plots can be rendered in environments without a display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    ):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.random_state = random_state


def generate_tsne_plot(
    features: np.ndarray,
    labels: Iterable[int],
    output_path: Optional[str] = None,
    config: Optional["TSNEConfig"] = None,
) -> Path:
    """
    Render a t-SNE scatter plot for the provided feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (samples x features) used during training.
    labels : Iterable[int]
        Binary labels aligned with the feature rows (0 = normal, 1 = attack).
    output_path : Optional[str]
        Where to save the resulting PNG. Defaults to ``tsne_scatter.png`` when
        not specified.
    config : Optional[TSNEConfig]
        Optional custom configuration for t-SNE hyperparameters.

    Returns
    -------
    pathlib.Path
        Path to the generated PNG file.
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
    logger.info(
        "Running t-SNE with perplexity=%s, learning_rate=%s, n_iter=%s",
        cfg.perplexity,
        cfg.learning_rate,
        cfg.n_iter,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=cfg.perplexity,
        learning_rate=cfg.learning_rate,
        n_iter=cfg.n_iter,
        metric=cfg.metric,
        random_state=cfg.random_state,
        init="pca",
        verbose=0,
    )

    embeddings = tsne.fit_transform(features)
    logger.debug("t-SNE embedding shape: %s", embeddings.shape)

    output = Path(output_path) if output_path else Path("tsne_scatter.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    is_attack = labels_array == 1
    plt.scatter(
        embeddings[~is_attack, 0],
        embeddings[~is_attack, 1],
        c="royalblue",
        alpha=0.6,
        label="Normal (0)",
        edgecolors="none",
    )
    plt.scatter(
        embeddings[is_attack, 0],
        embeddings[is_attack, 1],
        c="crimson",
        alpha=0.7,
        label="Attack (1)",
        edgecolors="none",
    )

    plt.title("t-SNE Projection of Training Windows")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("t-SNE scatter plot saved to %s", output)
    return output
