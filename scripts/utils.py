import os
import pickle as pkl
import sys

import numpy as np


def read_pickle_batches(file_name):
    with open(file_name, "rb") as f:
        try:
            while True:
                yield pkl.load(f)
        except EOFError:
            pass


def read_pickle(data_file):
    embeddings = None
    ids = []
    metadata = None
    for batch in read_pickle_batches(data_file):
        ids += batch["id"]
        bemb = batch["embeddings"]
        if not isinstance(embeddings, np.ndarray):
            embeddings = bemb
        else:
            embeddings = np.vstack((embeddings, bemb))
        if "metadata" in batch.keys():
            metadata = batch["metadata"]
    return ids, embeddings, metadata


def load_embeddings(data_file):
    ext = os.path.splitext(data_file)[1]
    if ext == ".pkl":
        return read_pickle(data_file)
    else:
        raise Exception("Only .pkl embedding file can be loaded")


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    Args
        filepath: the path to the file to be overwritten.

    Returns
        True if we can proceed with overwrite, False otherwise.
    """
    get_input = input
    # if sys.version_info[:2] <= (2, 7):
    #     get_input = raw_input
    overwrite = get_input("[WARNING] %s already exist - overwrite? " "[y/n]" % (filepath))
    while overwrite not in ["y", "n"]:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
    if overwrite == "n":
        return False
    print("[TIP] Next time specify --overwrite!")
    return True


def get_set_aws_credentials():
    credentials = input("Paste credentials file content:")
    while credentials == "":
        credentials = input("Paste credentials file content:")
    _, key, secret_key, token = credentials.split("\n")
    os.environ["aws_access_key_id"] = key.split("aws_access_key_id=")[1]
    os.environ["aws_secret_access_key"] = secret_key.split("aws_secret_access_key=")[1]
    os.environ["aws_session_token"] = token.split("aws_session_token=")[1]
