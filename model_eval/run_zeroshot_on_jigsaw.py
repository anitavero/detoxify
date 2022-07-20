import argparse
import json
import os

import eval_predictions

from scripts.run_classifiers import run_zeroshot


def test_zeroshot(config, data_path=None, device="cuda:0", eval=False, s3_dir=None):
    if data_path is None:
        data_path = config["dataset"]["args"]["test_csv_file"]

    prompt_pattern = "This text is about {}"

    model_names = config["arch"]["args"]["model_name"]
    if isinstance(model_names, str):
        model_names = [model_names]

    for model_name in model_names:
        print(model_name)
        files = run_zeroshot(
            data_path,
            candidate_labels=config["dataset"]["args"]["classes"],
            model_name=model_name,
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
        if eval:
            files["metrics"] = eval_predictions.save_metrics(files["results"], config)

        if s3_dir is not None:
            print("Copying to S3")
            os.system(f'aws s3 cp {files["results"]} {os.path.join(s3_dir, "results/")}')
            if eval:
                os.system(f'aws s3 cp {files["metrics"]} {os.path.join(s3_dir, "results/")}')
            os.system(f'aws s3 cp {files["embeddings"]} {os.path.join(s3_dir, "embeddings/")}')
            os.system(f'aws s3 cp {files["prompt_embeddings"]} {os.path.join(s3_dir, "embeddings/")}')


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

    results = test_zeroshot(config, args.test_csv, args.device, args.eval, args.s3_dir)
