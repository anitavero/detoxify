import argparse
import json

from scripts.run_classifiers import run_zeroshot


def test_zeroshot(config, data_path=None, device="cuda:0"):
    if data_path is None:
        data_path = config["dataset"]["args"]["test_csv_file"]

    run_zeroshot(
        data_path,
        model_name=config["arch"]["args"]["model_name"],
        embeddings_file=None,
        prompt_embeddings_file=None,
        batch_size=1,
        device=device,
        prompt_pattern="This text is about {}",
        candidate_labels=config["dataset"]["args"]["classes"],
        save_to="",
        save_embeddings_to="",
        overwrite=True,
        id_column="id",
        text_column="comment_text",
    )


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

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = test_zeroshot(config, args.test_csv, args.device)
