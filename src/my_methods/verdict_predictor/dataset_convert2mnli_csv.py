import os
import argparse
import h5py
import json
import pickle
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def claim2QAs(example):
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

            # yield prompt_str.replace("\n", " ").replace("||", "\n")
            yield ["[CLAIM] " + claim.strip(),
                   " [QUESTION] " + q_text.strip() + " " + a_text.strip(),
                   example["label"]]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()
    print(f"Provided arguments: {args}")

    cnt_samples = 0

    with open(os.path.join(args.data_dir, f"dev_train_mnli.tsv"), "w", encoding="utf-8") as f_out:
        for split in ['dev', 'train']:
            with open(os.path.join(args.data_dir, f"{split}.json"), "r", encoding="utf-8") as f_in:
                for line in json.load(f_in):
                    for cqa in claim2QAs(line):
                        claim, qa, averitec_label = cqa
                        claim = claim.replace('\n', '').replace('\t', '')
                        qa = qa.replace('\n','').replace('\t', '')
                        label = ""
                        if averitec_label == 'Refuted':
                            label = "contradiction"
                        elif averitec_label == 'Supported':
                            label = "entailment"
                        elif averitec_label == 'Not Enough Evidence':
                            label = "neutral"
                        else:
                            continue
                        averitec = """[CLAIM] Republican Matt Gaetz was part of a company that had to pay 75 million in hospice fraud. 
                        They stole from dying people.	 [QUESTION] What year did Don Gaetz sell Vitas to Chemed? In 2004, VITAS was 
                        acquired by Roto Rooter's parent company Chemed for $400 million.	Refuted"""

                        mnli = """dev	MNLI	multinli_1.0_dev_matched	This site includes a list of all award winners and a searchable database 
                        of Government Executive articles.	The Government Executive articles housed on the website are not able to be 
                        searched.	contradiction"""
                        assert label, print(f"Label should not be empty: {label}")
                        f_out.write(f"{split}\tAVERITEC\t{split}_1.0\t{qa}\t{claim}\t{label}\n")
                        cnt_samples += 1
    print(f"Samples created: {cnt_samples}")

