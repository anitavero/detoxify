import argparse

from utils import convert_pkls_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a pickle series file or a directory of pickle series files to webdataset format.")
    parser.add_argument(
        "data_path",
        type=str,
        help="File or directory path",
    )
    parser.add_argument(
        "--to",
        type=str,
        choices=['pickle', 'webdataset'],
        default='pickle',
        help="Format to convert to: 'pickle' or 'webdataset' (default: pickle)",
    )
    args = parser.parse_args()
    convert_pkls_embeddings(args.data_path, args.to)
