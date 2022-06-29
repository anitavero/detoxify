import os

import clip
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

import pandas as pd
from tqdm import tqdm
from typing import Union, List, Optional
import pickle as pkl
import re
import numpy as np


#### Zero shot models
    
class ZeroShotWrapper():
    """"
    Zeroshot models mased on pretrained embeddings. 
    We use the improved T5 versions from https://huggingface.co/docs/transformers/model_doc/t5v1.1.
    Args:
        model_name(str): model name to be loaded, can be
            bart
            roberta   
            t5-small
            t5-base
            t5-large
            t5-xl
            t5-xxl
     """
    def __init__(self, candidate_labels, hypothesis_template, model_name: Optional[str]="bart", device: Optional[str]="cpu"):
        self.model_name = {"bart": "facebook/bart-large-mnli", 
                           "roberta": "joeddav/xlm-roberta-large-xnli",
                           "t5-small": 'google/t5-v1_1-small',
                           "t5-base": 'google/t5-v1_1-base',
                           "t5-large": 'google/t5-v1_1-large',
                           "t5-xl": 'google/t5-v1_1-xl',
                           "t5-xxl": "google/t5-v1_1-xxl"}[model_name]
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.candidate_labels = candidate_labels
        self.hypotheses = [hypothesis_template.format(l) for l in candidate_labels]
        self.hypotheses_embeddings = self.model.encode(self.hypotheses, convert_to_tensor=True)
    
    def predict(self, input_text: Union[str, List[str]], embeddings=None) -> dict:
        if not isinstance(input_text, list):
            input_text = [input_text]
        if isinstance(embeddings, np.ndarray):
            input_embeddings = torch.from_numpy(embeddings).float().to(self.device)
        else:
            input_embeddings = self.model.encode(input_text, convert_to_tensor=True)
        predictions = util.pytorch_cos_sim(input_embeddings, self.hypotheses_embeddings).cpu()
        return {'predictions': {l: predictions[:,i] for (i, l) in enumerate(self.candidate_labels)},
                'embeddings': input_embeddings}

#### Linear probe trained over tranformer embeddings

class T5TextWrapper:
    def __init__(self, cla_path: str, model_name: Optional[str] = "sentence-transformers/sentence-t5-xl", device: Optional[str] = "cpu"):
        model_ckp = os.path.split(cla_path)[1]
        if not os.path.exists(model_ckp):
            os.system(f'aws s3 cp {cla_path} .')

        self.model = SentenceTransformer(model_name).to(device)
        self.model = self.model.eval()
        self.device = device
        self.classifier = pd.read_pickle(model_ckp)

    def predict(self, input_text: Union[str, List[str]]) -> dict:
        if not isinstance(input_text, list):
            input_text = [input_text]
        with torch.no_grad():
            text_feature = self.model.encode(input_text)
        predictions = self.classifier.predict_proba(text_feature)
        return {"toxicity": predictions[:, 0], "safety": predictions[:, 1]}


def run(model, text_data, embeddings=None, batch_size=1, col_names=['score'], save_name='results.csv', save_embs=False):

    def batch(your_list, bs=1):
        l = len(your_list)
        for i in range(0, l, bs):
            yield your_list[i:min(i + bs, l)]

    # save empty csv file where the results will be saved
    pd.DataFrame({'text': [], **{cl: [] for cl in col_names}}).to_csv(save_name, index=False)
    if save_embs:
        embf = open(re.sub('.csv', '_embeddings.pkl', save_name), "ab")
        pkl.dump({'prompts': model.hypotheses, 'embeddings': model.hypotheses_embeddings.cpu().numpy()}, embf, protocol=pkl.HIGHEST_PROTOCOL)

    cnt = 0
    if isinstance(embeddings, np.ndarray):
        for batched_text, batched_embeddings in tqdm(
            zip(batch(text_data.fillna("").to_list(), batch_size), batch(embeddings, batch_size)), 
            total=len(text_data) / batch_size):
            batched_result = model.predict(batched_text, batched_embeddings)
            if 'predictions' in batched_result.keys():
                dict_to_append = {'text': batched_text, **batched_result['predictions']}
            else:
                dict_to_append = {'text': batched_text, **batched_result}
            # append result dict for each batch to csv file
            pd.DataFrame(dict_to_append, index=list(range(cnt, cnt + min(batch_size, len(batched_text))))).to_csv(save_name,
                                                                                        mode='a', header=None, index=False)
            if save_embs:
                pkl.dump({'tweets': batched_text, 'embeddings': batched_result['embeddings'].cpu().numpy()}, embf, protocol=pkl.HIGHEST_PROTOCOL)

            cnt += batch_size
    else:
        for batched_text in tqdm(batch(text_data.fillna("").to_list(), batch_size), total=len(text_data) / batch_size):
            batched_result = model.predict(batched_text)
            if 'predictions' in batched_result.keys():
                dict_to_append = {'text': batched_text, **batched_result['predictions']}
            else:
                dict_to_append = {'text': batched_text, **batched_result}
            # append result dict for each batch to csv file
            pd.DataFrame(dict_to_append, index=list(range(cnt, cnt + min(batch_size, len(batched_text))))).to_csv(save_name,
                                                                                        mode='a', header=None, index=False)
            if save_embs:
                pkl.dump({'tweets': batched_text, 'embeddings': batched_result['embeddings'].cpu().numpy()}, embf, protocol=pkl.HIGHEST_PROTOCOL)

            cnt += batch_size

    if save_embs:    
        embf.close()