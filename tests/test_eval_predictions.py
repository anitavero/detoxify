import json

import numpy as np
import pandas as pd
import pytest

from model_eval.eval_predictions import evaluate, save_metrics


def test_evaluate():
    scores = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    res = evaluate(scores, targets)
    assert res["auc_scores"] == [1, 1, 1]
    assert res["mean_auc"] == 1

    scores = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0]])
    res = evaluate(scores, targets)
    assert res["auc_scores"] == [1, 1, 0.5]
    m = 2.5 / 3
    assert res["mean_auc"] == m

    scores = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    res = evaluate(scores, targets)
    assert res["auc_scores"] == [0.5, 0.5, 0.5]
    assert res["mean_auc"] == 0.5


def test_save_metrics(tmp_path):
    res_path = tmp_path / "tests/test_res.csv"
    label_path = tmp_path / "tests/test_labels.csv"
    test_path = tmp_path / "tests/test.csv"
    config_path = "tests/dummy_data/test_config.json"
    res_path.parent.mkdir()
    res_path.touch()
    test_path.touch()
    label_path.touch()

    with open(config_path) as f:
        config = json.load(f)

    pd.DataFrame({"id": [0, 1], "A": [0, 1], "B": [0, 1]}).to_csv(res_path, index=False)
    pd.DataFrame({"id": [0, 1], "C": [0, 1], "D": [0, 1]}).to_csv(label_path, index=False)
    pd.DataFrame({"id": [0, 1], "C": [0, 1], "D": [0, 1]}).to_csv(test_path, index=False)
    config["dataset"]["args"]["test_labels_csv_file"] = label_path
    config["dataset"]["type"] = "TEST"

    with pytest.raises(Exception, match="Classes in result and label file aren't matching"):
        save_metrics(res_path, config)

    config["dataset"]["args"]["classes"] = ["A", "B"]
    config["dataset"]["type"] = "JigsawDataOriginal"
    config["dataset"]["args"]["test_csv_file"] = str(test_path)

    with pytest.raises(Exception, match="Classes in result and label file aren't matching"):
        save_metrics(res_path, config)
