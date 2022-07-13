import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_loaders import JigsawDataBias, JigsawDataMultilingual, JigsawDataOriginal
from torch.utils.data import DataLoader


def evaluate(config, results_file, test_label_file):
    """Evaluate model prediction scores."""
    results = pd.read_csv(results_file, dtype={"id": "string"})
    data_loader = JigsawDataOriginal(
        test_csv_file="detoxify/jigsaw-toxic-comment-classification-challenge/data/test.csv",
        train=False,
    )
    labels = data_loader.load_data(test_label_file)

    classes = list(results.columns)
    classes.remove("id")
    assert results["id"].to_list() == labels["id"].to_list()
    scores = results[classes].to_numpy()
    targets = labels[classes].to_numpy()

    auc_scores = []
    for class_idx in range(scores.shape[1]):
        # labels for the test data; value of -1 indicates it was not used for scoring
        # from https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
        mask = targets[:, class_idx] != -1
        target_binary = targets[mask, class_idx]
        class_scores = scores[mask, class_idx]
        try:
            auc = roc_auc_score(target_binary, class_scores)
            auc_scores.append(auc)
        except Exception:
            warnings.warn(
                "Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
            )
            auc_scores.append(np.nan)

    mean_auc = np.mean(auc_scores)

    results = {
        "auc_scores": auc_scores,
        "mean_auc": mean_auc,
    }

    return results


def save_metrics(results_file, labels_file, config_file):
    results_name = os.path.splitext(os.path.basename(results_file))[0]
    dir = os.path.dirname(results_file)
    config = json.load(open(config_file))

    metrics = evaluate(config, results_file, labels_file)
    with open(os.path.join(dir, f"metrics_{results_name}.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model prediction scores.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--results",
        type=str,
        required=True,
        help="results file path",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        required=True,
        help="labels file path",
    )

    args = parser.parse_args()
    save_metrics(args.results, args.labels, args.config)
