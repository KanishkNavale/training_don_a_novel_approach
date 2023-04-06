from typing import List

import numpy as np
from matplotlib.pylab import plt

from src.datamodule import DataModule
from src.don import load_trained_don_model
from src.don.metrices import AUC_for_PCK
from src.utils import initialize_config_file
from src.configurations import DataLoaderConfig


def plot_metric(pcks: List[np.ndarray], aucs: List[np.ndarray]) -> None:

    pcks = np.vstack(pcks)
    min_pcks = np.min(pcks, axis=0)
    max_pcks = np.max(pcks, axis=0)
    mean_pcks = np.mean(pcks, axis=0)

    auc_mean = np.mean(np.hstack(aucs))
    auc_std = np.std(np.hstack(aucs))

    plt.plot(mean_pcks, 'r--')
    plt.fill_between(np.arange(len(mean_pcks)),
                     np.zeros_like(mean_pcks),
                     min_pcks,
                     alpha=0.05,
                     color="b",
                     hatch=".",
                     label="AUC")
    plt.fill_between(np.arange(len(mean_pcks)),
                     min_pcks,
                     max_pcks,
                     alpha=0.2,
                     color="r")

    plt.grid(True)
    plt.xlabel("K")
    plt.ylabel("PCK")
    plt.title(f"AUC for PCK@k, ∀k ∈ [1, 100] = {auc_mean:.4f}±{auc_std:.4f}")
    plt.legend(loc='best')
    plt.ylim(0, 1.25)

    plt.savefig("AUC_for_PCK.png")


if __name__ == "__main__":

    # Init. Trained DON Model
    trained_model = load_trained_don_model("sandbox/don/don-config.yaml")

    # Init. Dataloader
    config = initialize_config_file("sandbox/don/don-config.yaml")
    dataloader_config = DataLoaderConfig.from_dictionary(config)
    dataloader = DataModule(dataloader_config)

    PCKS, AUCS = AUC_for_PCK(trained_model,
                             dataloader,
                             iterations=2,
                             n_correspondences=150)

    plot_metric(PCKS, AUCS)
