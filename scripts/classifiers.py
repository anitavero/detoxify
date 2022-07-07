import os
import pickle as pkl
import re
from lib2to3.pgen2.token import OP
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .utils import ask_to_proceed_with_overwrite


class NShotWrapper:
    """
    Zeroshot or Fewshot models based on pretrained embeddings.
    We use the improved T5 versions from https://huggingface.co/docs/transformers/model_doc/t5v1.1.
    Args:
        candidate_labels(str list): label list
        model_name(str): model name to be loaded, can be
            bart
            roberta
            t5-small
            t5-base
            t5-large
            t5-xl
            t5-xxl
        device(str): device commpute on (default: cpu)
    """

    def __init__(
        self,
        candidate_labels,
        model_name: Optional[str] = "bart",
        device: Optional[str] = "cpu",
    ):
        self.model_name = {
            "bart": "facebook/bart-large-mnli",
            "roberta": "joeddav/xlm-roberta-large-xnli",
            "t5-small": "google/t5-v1_1-small",
            "t5-base": "google/t5-v1_1-base",
            "t5-large": "google/t5-v1_1-large",
            "t5-xl": "google/t5-v1_1-xl",
            "t5-xxl": "google/t5-v1_1-xxl",
        }[model_name]
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.candidate_labels = candidate_labels

    def encode(self, input_text: Union[str, List[str]], embeddings: Optional[np.ndarray] = None) -> dict:
        if not isinstance(input_text, list):
            input_text = [input_text]
        if isinstance(embeddings, np.ndarray):
            input_embeddings = torch.from_numpy(embeddings).float().to(self.device)
        else:
            input_embeddings = self.model.encode(input_text, convert_to_tensor=True)
        return input_embeddings


class ZeroShotWrapper(NShotWrapper):
    """
    Zeroshot models based on pretrained embeddings.
    Args:
        candidate_labels, model_name, device: see NShotWrapper
        hypothesis_template(str): Prompt pattern for zeroshot classification (has to include {})
        hypotheses_embeddings(np.ndarray): precomputer hypothesis embeddings
    """

    def __init__(
        self,
        candidate_labels,
        hypothesis_template: str,
        model_name: Optional[str] = "bart",
        device: Optional[str] = "cpu",
        hypotheses_embeddings: Optional[np.ndarray] = None,
    ):
        super().__init__(candidate_labels, model_name, device)
        self.hypotheses = [hypothesis_template.format(label) for label in candidate_labels]
        self.hypothesis_template = hypothesis_template
        if hypotheses_embeddings is not None:
            self.hypotheses_embeddings = torch.from_numpy(hypotheses_embeddings).float().to(self.device)
        else:
            self.hypotheses_embeddings = self.model.encode(self.hypotheses, convert_to_tensor=True)

    def predict(self, input_text: Union[str, List[str]], embeddings: Optional[np.ndarray] = None) -> dict:
        input_embeddings = self.encode(input_text, embeddings)
        predictions = util.pytorch_cos_sim(input_embeddings, self.hypotheses_embeddings).cpu()
        return {
            "predictions": {l: predictions[:, i] for (i, l) in enumerate(self.candidate_labels)},
            "embeddings": input_embeddings,
        }


# Linear probe trained over tranformer embeddings


# class T5TextWrapper:
#     def __init__(
#         self,
#         cla_path: str,
#         model_name: Optional[str] = "sentence-transformers/sentence-t5-xl",
#         device: Optional[str] = "cpu",
#     ):
#         model_ckp = os.path.split(cla_path)[1]
#         if not os.path.exists(model_ckp):
#             os.system(f"aws s3 cp {cla_path} .")

#         self.model = SentenceTransformer(model_name).to(device)
#         self.model = self.model.eval()
#         self.device = device
#         self.classifier = pd.read_pickle(model_ckp)

#     def predict(self, input_text: Union[str, List[str]]) -> dict:
#         if not isinstance(input_text, list):
#             input_text = [input_text]
#         with torch.no_grad():
#             text_feature = self.model.encode(input_text)
#         predictions = self.classifier.predict_proba(text_feature)
#         return {"toxicity": predictions[:, 0], "safety": predictions[:, 1]}


def run(
    model,
    dataset,
    embeddings=None,
    batch_size=1,
    col_names=["score"],
    save_name="results.csv",
    save_dir="",
    save_embeddings_to="",
    overwrite=False,
):
    text_data = dataset["text"].fillna("").to_list()
    ids = dataset["id"].to_list()

    def batch_gen(your_list, bs):
        ln = len(your_list)
        for i in range(0, ln, bs):
            yield your_list[i : min(i + bs, ln)]

    # save empty csv file where the results will be saved
    res_file = os.path.join(save_dir, f"results_{save_name}.csv")
    if os.path.exists(res_file) and overwrite is False:
        if not ask_to_proceed_with_overwrite(res_file):
            return False
    pd.DataFrame({"id": [], **{cl: [] for cl in col_names}}).to_csv(res_file, index=False)
    ext = {"pickle": "pkl"}[save_embeddings_to]  # For future extentions with further formats
    if save_embeddings_to:
        # open file for embedding batches
        emb_file = os.path.join(save_dir, f"embeddings_{save_name}.{ext}")
        is_zeroshot = isinstance(model, ZeroShotWrapper)
        if is_zeroshot:
            prompt_file = os.path.join(save_dir, f"prompt_embeddings_{save_name}.{ext}")
        else:
            prompt_file = ""
        exists = [emb_file if os.path.exists(emb_file) else None, prompt_file if os.path.exists(prompt_file) else None]
        if any(exists) and overwrite is False:
            overwrite = ask_to_proceed_with_overwrite(", ".join(filter(None, exists)))
            if not overwrite:
                return False
        if overwrite:
            for f in filter(None, exists):
                os.remove(f)
        if save_embeddings_to == "pickle":
            embf = open(emb_file, "ab")

        if is_zeroshot:
            # save prompts with embeddings
            if save_embeddings_to == "pickle":
                prompt_data = {
                    "id": model.candidate_labels,
                    "embeddings": model.hypotheses_embeddings.cpu().numpy(),
                    "metadata": model.hypothesis_template,
                }
                with open(prompt_file, "ab") as pr_embf:
                    pkl.dump(prompt_data, pr_embf, protocol=pkl.HIGHEST_PROTOCOL)

    cnt = 0
    if isinstance(embeddings, np.ndarray):
        batches = map(
            lambda x: {"id": x[0], "texts": x[1], "embeddings": x[2]},
            zip(batch_gen(ids, batch_size), batch_gen(text_data, batch_size), batch_gen(embeddings, batch_size)),
        )
    else:
        batches = map(
            lambda x: {"id": x[0], "texts": x[1], "embeddings": None},
            zip(batch_gen(ids, batch_size), batch_gen(text_data, batch_size)),
        )

    for batch in tqdm(batches, total=len(text_data) / batch_size):
        batched_result = model.predict(batch["texts"], batch["embeddings"])
        dict_to_append = {"id": batch["id"], **batched_result["predictions"]}
        # append result dict for each batch to csv file
        pd.DataFrame(dict_to_append, index=list(range(cnt, cnt + min(batch_size, len(batch))))).to_csv(
            res_file, mode="a", header=None, index=False
        )
        if save_embeddings_to:
            # append embeddings to embeddings file
            if save_embeddings_to == "pickle":
                emb_data = {"id": batch["id"], "embeddings": batched_result["embeddings"].cpu().numpy()}
                pkl.dump(emb_data, embf, protocol=pkl.HIGHEST_PROTOCOL)
        cnt += batch_size

    if save_embeddings_to == "pickle":
        embf.close()
