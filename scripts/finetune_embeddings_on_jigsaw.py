import argparse
import json
import os
from inspect import getfullargspec

import numpy as np

import pandas as pd
import wandb
import yaml
from model_eval.eval_predictions import evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

from utils import load_embeddings

from scripts.run_classifiers import run_zeroshot


def finetune(config, sweep_config=None, action="train", device="cuda:0", s3_dir=None):
    train_path = config["dataset"]["args"]["train_csv_file"]
    test_labels_path = config["dataset"]["args"]["test_labels_csv_file"]
    test_path = config["dataset"]["args"]["test_csv_file"]
    emb_dir = config["arch"]["args"]["embeddings_dir"]
    results_dir = config["arch"]["args"]["results_dir"]

    test_labels = pd.read_csv(test_labels_path, dtype={"id": "string"})
    train_labels = pd.read_csv(train_path, dtype={"id": "string"})
    classes = list(test_labels.columns)
    train_classes = list(train_labels.columns)
    train_classes.remove("comment_text")
    assert classes == train_classes
    classes.remove("id")

    classifier_name = config["arch"]["args"]["classifier"]

    model_names = config["arch"]["args"]["model_name"]
    if isinstance(model_names, str):
        model_names = [model_names]

    for model_name in model_names:
        print(model_name)
        files = []
        if model_name != "random":
            train_emb_file = os.path.join(emb_dir, f"embeddings_{model_name}_train.pkl")
            test_emb_file = os.path.join(emb_dir, f"embeddings_{model_name}_test.pkl")
            for data_path, pkl_path in [(train_path, train_emb_file), (test_path, test_emb_file)]:
                if not os.path.exists(pkl_path):
                    print("Save embeddings to", pkl_path)
                    files.append(
                        run_zeroshot(
                            data_path,
                            candidate_labels=classes,
                            model_name=model_name,
                            embeddings_file=None,
                            prompt_embeddings_file=None,
                            batch_size=config["batch_size"],
                            device=device,
                            prompt_pattern="This text is about {}",
                            save_to="",
                            save_embeddings_to="pickle",
                            overwrite=True,
                            id_column="id",
                            text_column="comment_text",
                        )
                    )
            print("Load embeddings")
            ids, train_embeddings, metadata = load_embeddings(train_emb_file)
            ids, test_embeddings, metadata = load_embeddings(test_emb_file)
        else:
            n_train = train_labels.shape[0]
            n_test = test_labels.shape[0]
            # t5-large 1024
            # t5-xl    2048
            # bart     1024
            # roberta  1024
            d = 1024
            seed = 2022
            rng = np.random.default_rng(seed)
            model_name += f"_d_{d}_seed_{seed}"
            train_embeddings = rng.random((n_train, d))
            test_embeddings = rng.random((n_test, d))

        y_test = test_labels[classes].to_numpy()
        y_train = train_labels[classes].to_numpy()

        mask = (y_test != -1).any(axis=1)
        y_test_m = y_test[mask, :]
        test_embeddings_m = test_embeddings[mask, :]

        if action == "train":
            train(
                classifier_name,
                train_embeddings,
                y_train,
                test_embeddings_m,
                y_test_m,
                results_dir,
                model_name,
                s3_dir,
                files,
            )
        if action == "sweep":
            sweep(
                sweep_config,
                classifier_name,
                train_embeddings,
                y_train,
                test_embeddings_m,
                y_test_m,
                results_dir,
                model_name,
            )


def train(
    classifier_name,
    train_embeddings,
    y_train,
    test_embeddings_m,
    y_test_m,
    results_dir,
    model_name,
    param_config,
    s3_dir=None,
    embedding_files=None,
    save_metrics=True,
):
    print("Train")

    def params(model):
        {p: val for p, val in param_config.items() if p in getfullargspec(model).args}

    if classifier_name == "mlp":
        clf = MLPClassifier(verbose=True, **params(MLPClassifier))
    elif classifier_name == "forest":
        clf = RandomForestClassifier(n_jobs=7, verbose=True, **params(RandomForestClassifier))

    mo_clf = MultiOutputClassifier(estimator=clf).fit(train_embeddings, y_train)

    def predict_prob(embs):
        preds = mo_clf.predict_proba(embs)
        return np.column_stack([p[:, 1] for p in preds])

    print("Predict")
    scores = predict_prob(test_embeddings_m)
    metrics = evaluate(scores, y_test_m)
    print("Mean Acc:", clf.score(test_embeddings_m, y_test_m))
    print(metrics)
    if save_metrics:
        metrics_file = os.path.join(results_dir, f"metrics_finetune_{model_name}_{classifier_name}_multioutput.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)

        if s3_dir is not None:
            print("Copying to S3")
            os.system(f'aws s3 cp {metrics_file} {os.path.join(s3_dir, "results/")}')
            for file in embedding_files:
                os.system(f'aws s3 cp {file["embeddings"]} {os.path.join(s3_dir, "embeddings/")}')

    return metrics


def sweep(
    sweep_config, classifier_name, train_embeddings, y_train, test_embeddings_m, y_test_m, results_dir, model_name
):
    with open(sweep_config) as stream:
        sweep_config = yaml.safe_load(stream)
    sweep_id = wandb.sweep(sweep_config)

    def run_train():
        wandb.init()  # required to have access to `wandb.config`
        wb_config = wandb.config
        metrics = train(
            classifier_name,
            train_embeddings,
            y_train,
            test_embeddings_m,
            y_test_m,
            results_dir,
            model_name,
            wb_config,
            save_metrics=False,
        )
        wandb.log({"mean_auc": metrics["mean_auc"]})

    count = 10  # number of runs to execute
    wandb.agent(sweep_id, function=run_train, count=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-a",
        "--action",
        default="train",
        type=str,
        choices=["train", "sweep"],
        help="Train models or sweep hyperparameters (default: train)",
    )
    parser.add_argument(
        "-sc",
        "--sweep_config",
        default=None,
        type=str,
        help="Sweep config file path (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "--s3_dir",
        default=None,
        type=str,
        help="save embeddings, results and metrics to the given s3 sirectory (default: None)",
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = finetune(config, args.sweep_config, args.action, args.device, args.s3_dir)
