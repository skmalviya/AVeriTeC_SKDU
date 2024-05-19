import random
from torch.utils.data.dataset import Dataset
import torch
from my_utils import load_jsonl_data, refine_obj_data
from tqdm import tqdm
from cleantext import clean
from urllib.parse import unquote
import unicodedata
from unidecode import unidecode
import re

pt = re.compile(r"\[\[.*?\|(.*?)]]")

def clean_text(text):
    if text != re.sub(pt, r"\1", text):
        d = 'shri'
    text = re.sub(pt, r"\1", text)
    text = unquote(text)
    text = unicodedata.normalize('NFD', text)
    text = clean(text.strip(),fix_unicode=True,               # fix various unicode errors
    to_ascii=False,                  # transliterate to closest ASCII representation
    lower=False,                     # lowercase text
    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=False,                # replace all email addresses with a special token
    no_phone_numbers=False,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
    )
    return text

class RobertaSentenceGenerator(Dataset):
    def __init__(self, input_path, tokenizer, cache_dir, data_type, args):
        super(RobertaSentenceGenerator, self).__init__()
        self.model_name = str(type(self))
        self.args = args
        self.config = args.config
        self.data_type = data_type
        self.raw_data = self.preprocess_raw_data(self.get_raw_data(input_path, keys=self.get_refine_keys()))
        if args.test_mode or (not "train" in self.data_type):
            self.cand_id_lst = self.get_cand_ids()

        self.tokenizer = args.tokenizer
        self.max_seq_len = 512
        self.generate_train_instances_one_epoch()

    def generate_train_instances_one_epoch(self):
        self.instances = []
        for entry in self.raw_data:
            if self.args.test_mode or (not "train" in self.data_type):
                for i, cand in enumerate(entry["neg_sents"]):
                    # first = entry["all_candidates"][i-1][0] if (i-1)>=0 else ""
                    # third = entry["all_candidates"][i+1][0] if (i+1)< len(entry["all_candidates"]) else ""
                    self.instances.append([unidecode(clean_text(entry["claim"])), unidecode(clean_text(cand)), None])
                    # self.instances.append([entry["claim"], cand[1] + " : " + first + " : " + cand[0] + " : " + third, None])
            else:
                pos_sents = entry["pos_sents"]
                neg_sents = entry["neg_sents"]
                if (not pos_sents) or (not neg_sents):
                    continue
                for ps in pos_sents*self.args.train_data_extend_multi:
                    # [sent, page_title, sent_id]
                    ns = random.choice(neg_sents)
                    self.instances.append([unidecode(clean_text(entry["claim"])), unidecode(clean_text(ps)), unidecode(clean_text(ns)) ])
        print("generate {} sentence pairs".format(len(self.instances)))

        if self.args.test_mode or (not "train" in self.data_type):
            assert len(self.instances) == sum([len(ci) for ci in self.cand_id_lst]) \
                , print(len(self.instances), sum([len(ci) for ci in self.cand_id_lst]))

    def get_cand_ids(self):
        cand_id_lst = []
        for entry in self.raw_data:
            cand_id_lst.append([cd for cd in entry["all_candidates"]])
        return cand_id_lst

    def print_example(self):
        pass
        # instance = self.raw_data[0]
        # for k, v in instance.items():
        #     print(k, " : ", v)
        #
        # instance = self.raw_data[-1]
        # for k, v in instance.items():
        #     print(k, " : ", v)

    def preprocess_raw_data(self, raw_data):
        return raw_data

    def get_refine_keys(self):
        keys = None
        return keys

    def get_raw_data(self, input_path, keys=None):
        raw_data = load_jsonl_data(input_path)
        if keys is not None:
            raw_data = refine_obj_data(raw_data, keys)
        return raw_data

    def get_encodings(self, s1, s2):
        pad_idx = self.tokenizer.pad_token_id
        max_len = self.max_seq_len

        if s2 is None:
            input_ids = [2,2]
            input_mask = [1,1]
        else:
            encodes = self.tokenizer(s1, s2)
            input_ids = encodes["input_ids"][:self.max_seq_len]
            input_mask = [1] * len(input_ids)
            input_ids += [pad_idx] * (max_len - len(input_ids))
            input_mask += [0] * (max_len - len(input_mask))
        return input_ids, input_mask

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # raw_data = self.raw_data[idx]
        instance = self.instances[idx]
        pos_ids, pos_attention_mask = self.get_encodings(instance[0], instance[1])
        pos_ids = torch.tensor(pos_ids).to(self.args.device)
        pos_attention_mask = torch.tensor(pos_attention_mask).to(self.args.device)

        neg_ids, neg_attention_mask = self.get_encodings(instance[0], instance[2])
        neg_ids = torch.tensor(neg_ids).to(self.args.device)
        neg_attention_mask = torch.tensor(neg_attention_mask).to(self.args.device)

        return pos_ids, pos_attention_mask, neg_ids, neg_attention_mask
