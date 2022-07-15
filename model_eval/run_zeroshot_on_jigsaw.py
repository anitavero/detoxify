import argparse
import json
import os

import eval_predictions

from scripts.run_classifiers import run_zeroshot


def test_zeroshot(config, data_path=None, device="cuda:0", eval=False):
    if data_path is None:
        data_path = config["dataset"]["args"]["test_csv_file"]

    prompt_pattern = "This text is about {}"

    run_zeroshot(
        data_path,
        candidate_labels=config["dataset"]["args"]["classes"],
        model_name=config["arch"]["args"]["model_name"],
        embeddings_file=None,
        prompt_embeddings_file=None,
        batch_size=config["batch_size"],
        device=device,
        prompt_pattern=prompt_pattern,
        save_to="",
        save_embeddings_to="pickle",
        overwrite=True,
        id_column="id",
        text_column="comment_text",
    )

    print("Evaluate")
    dataset_name = os.path.basename(data_path).split(".")[0]
    save_name = f'results_{config["arch"]["args"]["model_name"]}_{dataset_name}_{"_".join(prompt_pattern.split())}.csv'
    save_dir = os.path.dirname(data_path)
    results_file = os.path.join(save_dir, save_name)
    if eval:
        eval_predictions.save_metrics(results_file, config)


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
        "-t",
        "--test_csv",
        default=None,
        type=str,
        help="path to test dataset",
    )
    parser.add_argument(
        "--eval",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="save evaluation metrics (default: False)",
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = test_zeroshot(config, args.test_csv, args.device, args.eval)
