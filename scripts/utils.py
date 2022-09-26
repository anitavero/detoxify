import os
import pickle as pkl
import re
from email import message
from glob import glob

import numpy as np
import webdataset as wds


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


def pickles2webdataset(data_file):
    """Convert a pickle series file to webdataset format."""
    ids, embeddings, metadata = read_pickle(data_file)  # TODO: store metadata
    sink = wds.TarWriter(re.sub(".pkl", ".tar", data_file))
    for id, emb in zip(ids, embeddings):
        sink.write({"__key__": id, "embedding.pyd": emb})
    sink.close()


def embeddings2pkl(data_file):
    """Convert a pickle series file to merged pickle format."""
    ids, embeddings, metadata = read_pickle(data_file)
    with open(re.sub(".pkl", "_merged.pkl", data_file), "wb") as f:
        pkl.dump({"ids": ids, "embeddings": embeddings, "metadata": metadata}, f)


def convert_pkls_embeddings(data_path, to="pickle", file_pattern="*"):
    """Convert a pickle series file or a directory of pickle serise files to pickle or webdataset format."""
    if to == "pickle":
        convert = embeddings2pkl
    elif to == "webdataset":
        convert = pickles2webdataset
    else:
        raise Exception("Can only be converted to 'pickle' or 'webdataset'.")

    if os.path.isdir(data_path):
        for file in glob(os.path.join(data_path, f"*{file_pattern}*.pkl")):
            print("Converting", file)
            convert(file)
    else:
        print("Converting", data_path)
        convert(data_path)


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
