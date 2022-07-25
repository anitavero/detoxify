import argparse
import json
import os

import numpy as np

import pandas as pd
from model_eval.eval_predictions import evaluate
from sklearn.neural_network import MLPClassifier

from utils import load_embeddings

from scripts.run_classifiers import run_zeroshot


def finetune(config, device="cuda:0", s3_dir=None):
    train_path = config["dataset"]["args"]["train_csv_file"]
    test_labels_path = config["dataset"]["args"]["test_labels_csv_file"]
    test_path = config["dataset"]["args"]["test_csv_file"]
    data_dir = os.path.dirname(train_path)

    test_labels = pd.read_csv(test_labels_path, dtype={"id": "string"})
    train_labels = pd.read_csv(train_path, dtype={"id": "string"})
    classes = list(test_labels.columns)
    train_classes = list(train_labels.columns)
    train_classes.remove("comment_text")
    assert classes == train_classes
    classes.remove("id")

    model_names = config["arch"]["args"]["model_name"]
    if isinstance(model_names, str):
        model_names = [model_names]

    for model_name in model_names:
        print(model_name)
        files = []
        if model_name != "random":
            train_emb_file = os.path.join(data_dir, f"embeddings_{model_name}_train.pkl")
            test_emb_file = os.path.join(data_dir, f"embeddings_{model_name}_test.pkl")
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

        clf = MLPClassifier(random_state=1, max_iter=300, verbose=True).fit(train_embeddings, y_train)

        scores = clf.predict_proba(test_embeddings_m)
        metrics = evaluate(scores, y_test_m)
        print("Mean Acc:", clf.score(test_embeddings_m, y_test_m))
        print(metrics)
        metrics_file = os.path.join(data_dir, f"metrics_finetune_{model_name}.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)

        if s3_dir is not None:
            print("Copying to S3")
            os.system(f'aws s3 cp {metrics_file} {os.path.join(s3_dir, "results/")}')
            for file in files:
                os.system(f'aws s3 cp {file["embeddings"]} {os.path.join(s3_dir, "embeddings/")}')


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

    results = finetune(config, args.device, args.s3_dir)
