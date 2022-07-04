from lib2to3.pgen2.token import OP
import os

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
from utils import ask_to_proceed_with_overwrite


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
    def __init__(self, candidate_labels, hypothesis_template: str, model_name: Optional[str]="bart", 
                 device: Optional[str]="cpu", hypotheses_embeddings: Optional[np.ndarray]=None):
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
        self.hypothesis_template = hypothesis_template
        if hypotheses_embeddings is not None:
            self.hypotheses_embeddings = torch.from_numpy(hypotheses_embeddings).float().to(self.device)
        else:
            self.hypotheses_embeddings = self.model.encode(self.hypotheses, convert_to_tensor=True)
    
    def predict(self, input_text: Union[str, List[str]], embeddings:Optional[np.ndarray]=None) -> dict:
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


def run(model, dataset, embeddings=None, batch_size=1, col_names=['score'], save_name='results.csv', save_dir='', save_embs=False, overwrite=False):
    text_data = dataset['text'].fillna('').to_list()
    ids = dataset['id'].to_list()

    def batch_gen(your_list, bs):
        l = len(your_list)
        for i in range(0, l, bs):
            yield your_list[i:min(i + bs, l)]

    # save empty csv file where the results will be saved
    res_file = os.path.join(save_dir, f'results_{save_name}.csv') 
    if os.path.exists(res_file) and overwrite == False:
        if not ask_to_proceed_with_overwrite(res_file):
            return False
    pd.DataFrame({'id': [], **{cl: [] for cl in col_names}}).to_csv(res_file, index=False)
    if save_embs:
        # open file for embedding batches
        emb_file = os.path.join(save_dir, f'embeddings_{save_name}.pkl')
        prompt_file = os.path.join(save_dir, f'prompt_embeddings_{save_name}.pkl')
        exists = [emb_file if os.path.exists(emb_file) else None, prompt_file if os.path.exists(prompt_file) else None]
        if any(exists) and overwrite == False:
            overwrite = ask_to_proceed_with_overwrite(', '.join(filter(None, exists)))
            if not overwrite:
                return False
        if overwrite:
            for f in filter(None, exists):
                os.remove(f)

        embf = open(emb_file, "ab")
        # save prompts with embeddings
        pr_embf = open(prompt_file, "ab")
        pkl.dump({'ids': model.candidate_labels, 
                  'embeddings': model.hypotheses_embeddings.cpu().numpy(),
                  'metadata': model.hypothesis_template}, 
                  pr_embf, protocol=pkl.HIGHEST_PROTOCOL)
        pr_embf.close()

    cnt = 0
    if isinstance(embeddings, np.ndarray):
        batches = map(lambda x: {'ids': x[0], 'texts': x[1], 'embeddings': x[2]}, 
                        zip(batch_gen(ids, batch_size), batch_gen(text_data, batch_size), batch_gen(embeddings, batch_size)))
    else:
        batches = map(lambda x: {'ids': x[0], 'texts': x[1], 'embeddings': None},
                        zip(batch_gen(ids, batch_size), batch_gen(text_data, batch_size)))

    for batch in tqdm(batches, total=len(text_data) / batch_size):
        batched_result = model.predict(batch['texts'], batch['embeddings'])
        dict_to_append = {'ids': batch['ids'], **batched_result['predictions']}
        # append result dict for each batch to csv file
        pd.DataFrame(dict_to_append, index=list(range(cnt, cnt + min(batch_size, len(batch))))).to_csv(res_file,
                                                                                    mode='a', header=None, index=False)
        if save_embs:
            # append embeddings to embeddings file
            pkl.dump({'ids': batch['ids'], 'embeddings': batched_result['embeddings'].cpu().numpy()}, embf, protocol=pkl.HIGHEST_PROTOCOL)
        cnt += batch_size

    if save_embs:    
        embf.close()