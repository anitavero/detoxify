import argparse

from utils import pickles2webdataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a pickle series file or a directory of pickle series files to webdataset format.")
    parser.add_argument(
        "data_path",
        type=str,
        help="File or directory path",
    )
    args = parser.parse_args()
    pickles2webdataset(args.data_path)
