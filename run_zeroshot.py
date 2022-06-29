import os

import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.rcParams["savefig.dpi"] = 200
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
import pickle as pkl
import argparse

from detoxify import Detoxify
from classifiers import run, ZeroShotWrapper


def read_pickle_batches(file_name):
    with open(file_name, 'rb') as f:
        try:
            while True:
                yield pkl.load(f)
        except EOFError:
            pass

def load_embeddings(data_file):
    embeddings = None
    prompt_embeddings = None
    texts = []
    prompts = []
    for batch in read_pickle_batches(data_file):
        if 'prompts' in batch.keys():
            prompts = batch['prompts']
            prompt_embeddings = batch['embeddings']
        else:
            texts += batch['tweets']
            bemb = batch['embeddings']
            if not isinstance(embeddings, np.ndarray):
                embeddings = bemb
            else:
                embeddings = np.vstack((embeddings, bemb))
    return prompts, prompt_embeddings, texts, embeddings


def run_zeroshot(data_path, embeddings_file=None, model_name='bart', batch_size=1, device='cpu', 
                  prompt_pattern='This text is about {}', save_to='', save_embeddings=False):
    dataset = pd.read_csv(data_path, sep=",")
    text_col, id_col = get_cols_from_filename(data_path)
    if prompt_pattern:
        save_name=f'results_twitter_{text_col}_{model_name}_prm_{"_".join(prompt_pattern.split())}.csv'
    else:
        save_name=f'results_twitter_{text_col}_{model_name}.csv'
    if save_to:
        save_name = re.sub('.csv', f'_{save_to}.csv', save_name)
    candidate_labels = list(GARM_FINE_LABELS.values())
    print(prompt_pattern)
    print('Labels:\n', '\n'.join(candidate_labels))
    model = ZeroShotWrapper(candidate_labels, prompt_pattern, model_name, batch_size, device)

    if embeddings_file:
        print('Load embeddings')
        if model_name == 't5xxl':
            df = pd.read_pickle('s3://unitary.ai.datasets/twitter/text_embeddings/twitter_embeddings_t5_xxl.pkl')
            embeddings = df['embeddings'].to_numpy()
        else:
            prompts, prompt_embeddings, texts, embeddings = load_embeddings(embeddings_file)
    else:
        embeddings = None
    run(model, dataset[text_col].fillna(''), embeddings=embeddings, batch_size=batch_size, col_names=candidate_labels, save_name=save_name, save_embs=save_embeddings)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to dataset csv file",
    )
    parser.add_argument(
        "--model_name",
        default="t5-small",
        type=str,
        choices=['bart', 'roberta', 't5-small', 't5-base', 't5-large', 't5-xl', 't5-xxl'],
        help="Name of the model (default: t5-small)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to load the model on (default cpu)",
    )
    parser.add_argument(
        "--embeddings_file",
        default=None,
        type=str,
        help="path to pretrained embeddings to load",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="destination path to output model results to (default: results_<dataset>)",
    )
    parser.add_argument(
        "--save_embeddings",
        default=False,
        type=bool,
        help="save embeddings to <save_to>_embeddings or <dataset>_embeddings", #TODO: format options
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="size of test data batches."
    )
    parser.add_argument(
        "--prompt_pattern",
        default='This text is about {}',
        type=str,
        help="Prompts pattern to query classes. (default: This text is about {})",
    )

    args = parser.parse_args()

    run_zeroshot(
        args.data_path,
        args.embeddings_file,
        args.model_name,
        args.batch_size,
        args.save_to,
        args.device,
        args.prompt_pattern,
        args.save_to,
        args.save_embeddings
    )