import argparse
import json
import os
from glob import glob


def copy_to_s3(file_pattern, s3_config="scripts/s3_config.json", embeddings=True, metrics=True, results=False):
    with open(s3_config) as f:
        config = json.load(f)
    files = glob(os.path.join(config["local_dir"], f"*{file_pattern}*"))
    if files == []:
        files = glob(os.path.join(config["local_dir"], f"*/*{file_pattern}*"))
    if embeddings:
        for file in filter(lambda x: "embeddings" in x, files):
            os.system(f'aws s3 cp {file} {os.path.join(config["s3_dir"], "embeddings/")}')
    if metrics:
        for file in filter(lambda x: "metrics" in x, files):
            os.system(f'aws s3 cp {file} {os.path.join(config["s3_dir"], "results/")}')
    if results:
        for file in filter(lambda x: "results" in x, files):
            os.system(f'aws s3 cp {file} {os.path.join(config["s3_dir"], "results/")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--s3_config",
        default="scripts/s3_config.json",
        type=str,
        help="config file path (default: s3_config.json)",
    )
    parser.add_argument(
        "-p",
        "--file_pattern",
        required=True,
        type=str,
        help="Pattern for files to copy",
    )
    parser.add_argument(
        "--embeddings",
        default=True,
        type=bool,
        help="Copy embeddings to the given s3 directory (default: True)",
    )
    parser.add_argument(
        "--metrics",
        default=True,
        type=bool,
        help="Copy metrics to the given s3 directory (default: True)",
    )
    parser.add_argument(
        "--results",
        default=False,
        type=bool,
        help="Copy results to the given s3 directory (default: False)",
    )

    args = parser.parse_args()

    copy_to_s3(
        file_pattern=args.file_pattern,
        s3_config=args.s3_config,
        embeddings=args.embeddings,
        metrics=args.metrics,
        results=args.results,
    )
