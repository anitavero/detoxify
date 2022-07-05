import argparse
import os
import pickle as pkl
import warnings

import numpy as np

import pandas as pd

from classifiers import run, ZeroShotWrapper


DATASET_DESCRIPTION = (
    'The first two columns of the datset has to be "id" and "text", '
    + "there can be further columns including classnames:\n<id>,<text>[,<class_1>...<class_n>]"
)


def read_pickle_batches(file_name):
    with open(file_name, "rb") as f:
        try:
            while True:
                yield pkl.load(f)
        except EOFError:
            pass


def load_embeddings(data_file):
    embeddings = None
    ids = []
    metadata = None
    for batch in read_pickle_batches(data_file):
        ids += batch["ids"]
        bemb = batch["embeddings"]
        if not isinstance(embeddings, np.ndarray):
            embeddings = bemb
        else:
            embeddings = np.vstack((embeddings, bemb))
        if "metadata" in batch.keys():
            metadata = batch["metadata"]
    return ids, embeddings, metadata


def run_zeroshot(
    data_path,
    model_name="t5-small",
    embeddings_file=None,
    prompt_embeddings_file=None,
    batch_size=1,
    device="cpu",
    prompt_pattern="This text is about {}",
    candidate_labels=None,
    save_to="",
    save_embeddings=False,
    overwrite=False,
):
    dataset_name = os.path.basename(data_path).split(".")[0]
    dataset = pd.read_csv(data_path, sep=",", dtype={"id": "string"})  # read ids as strings so they don't get messed up
    columns = list(dataset.columns)
    if len(columns) < 2 or columns[0] != "id" or columns[1] != "text":
        raise Exception(DATASET_DESCRIPTION)
    if save_to:
        save_name = os.path.splitext(os.path.basename(save_to))[0]
        save_dir = os.path.dirname(save_to)
    else:
        save_name = f'{model_name}_{dataset_name}_{"_".join(prompt_pattern.split())}'
        save_dir = os.path.dirname(data_path)
    if candidate_labels is None:
        if len(columns) > 2:
            candidate_labels = columns[2:]
        else:
            raise Exception(
                f"Either candidate_labels should be given or {data_path} should include classnames as columns [2:]"
            )

    print("Prompt pattern:", prompt_pattern)
    print("Labels:")
    print("\n".join(candidate_labels))

    if embeddings_file:
        print("Load embeddings")
        ids, embeddings, _ = load_embeddings(embeddings_file)
    else:
        embeddings = None
    if prompt_embeddings_file:
        print("Load prompt embeddings")
        loaded_candidate_labels, prompt_embeddings, loaded_prompt_pattern = load_embeddings(prompt_embeddings_file)
        if loaded_candidate_labels != candidate_labels:
            raise Exception(
                "Prompt embedding labels are not matching with given labels:\n"
                + f"Prompt embedding:{loaded_candidate_labels}\nGiven labels:{candidate_labels}"
            )
        if loaded_prompt_pattern != prompt_pattern:
            warnings.warn(
                "Loaded prompt embedding pattern is not matching with given pattern."
                + f"We use the loaded one:\n{loaded_prompt_pattern}"
            )
            prompt_pattern = loaded_prompt_pattern
    else:
        prompt_embeddings = None

    model = ZeroShotWrapper(candidate_labels, prompt_pattern, model_name, device, prompt_embeddings)
    run(
        model,
        dataset,
        embeddings=embeddings,
        batch_size=batch_size,
        col_names=candidate_labels,
        save_name=save_name,
        save_dir=save_dir,
        save_embs=save_embeddings,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to dataset csv file. " + DATASET_DESCRIPTION,
    )
    parser.add_argument(
        "--model_name",
        default="t5-small",
        type=str,
        choices=["bart", "roberta", "t5-small", "t5-base", "t5-large", "t5-xl", "t5-xxl"],
        help="Name of the model (default: t5-small)",
    )
    parser.add_argument(
        "--embeddings_file",
        default=None,
        type=str,
        help="path to pretrained embeddings to load",
    )
    parser.add_argument(
        "--prompt_embeddings_file",
        default=None,
        type=str,
        help="path to pretrained prompt embeddings to load",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="size of test data batches.")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to load the model on (default cpu)",
    )
    parser.add_argument(
        "--prompt_pattern",
        default="This text is about {}",
        type=str,
        help="Prompt pattern to query classes. (default: 'This text is about {}"
        + " -- classnames are placed inside the {})",
    )
    parser.add_argument(
        "--candidate_labels",
        default=None,
        type=list,
        help="Candidate labels inserted in prompt_pattern."
        + " If None, labels will be read from the dataset columns [2:] (default: None)",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="destination path to output model results to"
        + " (default: results_<model_name>_<dataset>_<prompt_pattern>.csv)",
    )
    parser.add_argument(
        "--save_embeddings",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="save embeddings to embeddings_<save_to> or embeddings_<model_name>_<dataset>_<prompt_pattern>.pkl",
        # TODO: format options
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="overwrite existing results and embeddings files (default: False)",
    )

    args = parser.parse_args()

    run_zeroshot(
        args.data_path,
        args.model_name,
        args.embeddings_file,
        args.prompt_embeddings_file,
        args.batch_size,
        args.device,
        args.prompt_pattern,
        args.candidate_labels,
        args.save_to,
        args.save_embeddings,
        args.overwrite,
    )
