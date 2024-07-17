import os
import h5py
import json
import pickle
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger()

def claim2prompts(example):
    claim = example["claim"]
    if example['speaker'] == None or example['speaker'] == '':
        example['speaker'] = 'Speaker'
    claim_str = f"Outrageously, {example['speaker']} Claim: \"" + claim + "\"|| with Evidence:"

    for question in example["questions"]:
        q_text = question["question"].strip()
        if len(q_text) == 0:
            continue

        if not q_text[-1] == "?":
            q_text += "?"

        answer_strings = []

        for a in question["answers"]:
            if a["answer_type"] in ["Extractive", "Abstractive"]:
                answer_strings.append(a["answer"])
            if a["answer_type"] == "Boolean":
                answer_strings.append(
                    a["answer"]
                    + ", because "
                    + a["boolean_explanation"].lower().strip()
                )

        for a_text in answer_strings:
            if not a_text[-1] in [".", "!", ":", "?"]:
                a_text += "."

            prompt_str = (
                    claim_str + " " + a_text.strip() + "|| as an answer to the Question: " + q_text
                # claim_str + " " + a_text.strip() + "||Question answered: " + q_text
            )

            yield prompt_str.replace("\n", " ").replace("||", "\n")



class FinetuneData:
    r"""
    :returns
        examples: List of str=Prompt_Response extracted from split.json
    """

    def __init__(self, root='data', split='train'):
        self.file_path = os.path.join(root, split+'.json')

        with open(self.file_path, "r", encoding="utf-8") as json_file:
            examples = json.load(json_file)

        prompt_samples = []
        logger.info(f'Loading {split} dataset...')
        for example in examples:
            for prompt in claim2prompts(example):
                prompt_samples.append(prompt)
        self.prompt_samples = prompt_samples
        logger.info(f'Total number of {split} samples: {len(self.prompt_samples)}')



    def filter_data(self):
        # remove samples without positive in training
        if self.split == 'train':
            unlabeled_qs = [q for q in self.data if self.data[q]['label'].sum() == 0]
            for q in unlabeled_qs:
                self.data.pop(q)

    def prepare_features(self):
        with h5py.File(os.path.join(self.root, self.split + '_scores.hdf5'), 'r') as f:
            for q in tqdm(self.data):
                bm25_matrix = torch.FloatTensor(np.concatenate((f[q]['query_ctx_bm25_score'][()],
                                                                f[q]['ctx_ctx_bm25_score'][()]), axis=0))
                bm25_matrix = bm25_matrix[:int(1 + self.list_len), :int(self.num_anchors)]
                bm25_matrix = self.norm_feature(bm25_matrix, 100)
                dense_matrix = torch.FloatTensor(np.concatenate((f[q]['query_ctx_dense_score'][()],
                                                                 f[q]['ctx_ctx_dense_score'][()]), axis=0))
                dense_matrix = dense_matrix[:int(1 + self.list_len), :int(self.num_anchors)]
                dense_matrix = self.norm_feature(dense_matrix, 10)
                self.data[q]['feature'] = torch.stack([bm25_matrix, dense_matrix], dim=0)

    @staticmethod
    def norm_feature(x, norm_temperature=None):
        if norm_temperature is not None:
            x = (x / norm_temperature).softmax(dim=-1)

        # max-min normalization
        norm_min = x.min(dim=-1, keepdim=True).values
        norm_max = x.max(dim=-1, keepdim=True).values
        x = (x - norm_min) / (norm_max - norm_min + 1e-10)
        x = x * 2.0 - 1
        return x

    def prepare_labels(self):
        for q, d in self.data.items():
            if 'has_answer' in d and self.split != 'train':
                # for NQ evaluation
                has_answer = d['has_answer']
                label = [hit for hit in has_answer]
            else:
                positive_ctxs = d['positive_ctxs']
                retrieved_ctxs = d['retrieved_ctxs']
                label = [pid in positive_ctxs for pid in retrieved_ctxs]
            self.data[q]['label'] = torch.BoolTensor(label)[:int(self.list_len)]

    def __getitem__(self, index):
        q = self.qs[index]
        sim_matrix = self.data[q]['feature']
        label = self.data[q]['label']

        if self.shuffle_retrieval_list:
            label_perm_idx = torch.randperm(label.shape[0])
            label = label[label_perm_idx]

            matrix_perm_idx = torch.cat((torch.zeros(1, dtype=torch.long),
                                         label_perm_idx + torch.scalar_tensor(1, dtype=torch.long)))
            sim_matrix = sim_matrix[matrix_perm_idx]
            if self.perm_feat_dim:
                sim_matrix = sim_matrix[:, matrix_perm_idx]

        return q, sim_matrix, label

    def __len__(self):
        return len(self.qs)
