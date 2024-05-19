import os
import json
import torch
from tqdm import tqdm

# from baseline.retriever.eval_sentence_retriever import eval_sentence_obj
from roberta_sentence_arg_parser import get_parser

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from roberta_sentence_preprocessor import BasePreprocessor
from my_utils import set_seed, save_jsonl_data

from roberta_cls import RobertaCls
from roberta_sentence_generator import RobertaSentenceGenerator
import graph_cell_config as config

from prediction.evaluate_veracity import AVeriTeCEvaluator


def main(args):
    if args is None:
        args = get_parser().parse_args()
    assert args.test_ckpt is not None
    set_seed(args.seed)

    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    preprocessor = BasePreprocessor(args)

    args.label2id = config.label2idx
    args.id2label = dict(zip([value for _, value in args.label2id.items()], [key for key, _ in args.label2id.items()]))
    args.config = config
    args.tokenizer = tokenizer

    if args.model_type == "RobertaCls":
        model = RobertaCls(args)
        data_generator = RobertaSentenceGenerator
    else:
        assert False, args.model_type

    args.test_mode = True

    # train_data, valid_data, test_data = preprocessor.process(
    #     args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["dev"])
    # train_data, valid_data, test_data = preprocessor.process(
    #     args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["dev", "train", "test"])
    train_data, valid_data, test_data = preprocessor.process(
        args.data_dir, args.cache_dir, data_generator, args.tokenizer, dataset=["dev"])

    load_model_path = os.path.join(args.ckpt_root_dir, args.test_ckpt)

    ckpt_meta = model.load(load_model_path)
    model.to(args.device)

    if valid_data:
        dev_dataloader = DataLoader(valid_data, args.batch_size, shuffle=False)
        val(model, dev_dataloader, "dev", args)
    if train_data:
        train_dataloader = DataLoader(train_data, args.batch_size, shuffle=False)
        val(model, train_dataloader, "train", args)
    if test_data:
        test_dataloader = DataLoader(test_data, args.batch_size, shuffle=False)
        val(model, test_dataloader, "test", args)

@torch.no_grad()
def val(model, dataloader, split, args):
    """
    计算模型在验证集上的准确率等信息
    """
    dataloader.dataset.print_example()

    output_path = os.path.join(args.data_dir, f"{split}.sentences.roberta.s100.jsonl" + "." + args.test_ckpt)
    print(f"save sentence scores to {output_path}")

    preds_epoch = []

    model.eval()
    for ii, data_entry in tqdm(enumerate(dataloader)):
        res = model(data_entry, args, test_mode=True)
        score_pos, _, golds = res
        preds = list(score_pos.cpu().detach().numpy())
        preds_epoch.extend(preds)

    cand_id_lst = dataloader.dataset.cand_id_lst
    assert len(preds_epoch) == sum([len(ci) for ci in cand_id_lst]), print(len(preds_epoch), sum([len(ci) for ci in cand_id_lst]))
    pred_ids = []
    sent_scores = []
    stat = 0
    preds_epoch = [float(p) for p in preds_epoch]
    for cand_ids in cand_id_lst:
        pred_scores = preds_epoch[stat:stat+len(cand_ids)]
        preds = list(torch.topk(torch.tensor(pred_scores), k=min(200, len(pred_scores)))[1].numpy())
        sent_scores.append(pred_scores)
        pred_ids.append(preds)
        stat += len(cand_ids)

    predictions = []
    with open(output_path, "w", encoding="utf-8") as output_json:
        for entry, preds, sent_score in zip(dataloader.dataset.raw_data, pred_ids, sent_scores):
            oentry = {
            "claim_id": entry["id"],
            "claim": entry["claim"],
            "top_100": [{**entry["top_100"][idx], **{"score":sent_score[idx]}} for idx in preds]
            }

            predictions.append([e["sentence"] for e in oentry["top_100"]])
            output_json.write(json.dumps(oentry, ensure_ascii=False) + "\n")
            output_json.flush()
        

    references = [e['pos_sents'] for e in dataloader.dataset.raw_data]
    scorer = AVeriTeCEvaluator()
    valid_scores = []
    for level in [5, 10, 50, 100]:
        score = scorer.evaluate_src_tgt(predictions, references, max_sent=level)
        print(f"Answer-only score metric=(HU-{scorer.metric}) level={level} : {score}")
        valid_scores.append(score)

    return valid_scores[1]

if __name__ == '__main__':
    main(None)
