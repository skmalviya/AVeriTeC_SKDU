import argparse
import json

def generate_lines(line, gold_line, split):
    #print(f'Line:{line}')

    id = line["claim_id"]
    claim = gold_line["claim"]

    pos_sents = []

    if split != 'test':
        for question in gold_line["questions"]:
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
                pos_sents.append(a_text)


    neg_sents = [s["sentence"] for s in line["top_100"]]

    oentry = {
        "id": id,
        "claim": claim,
        "pos_sents": pos_sents,
        "neg_sents": neg_sents,
        "all_candidates": pos_sents + neg_sents if split=='train' else neg_sents,
        "top_100": line["top_100"],
    }

    return oentry

if __name__ == "__main__":
    # LogHelper.setup()
    # LogHelper.get_logger(__name__)

    parser = argparse.ArgumentParser(
        description="Get top 100 sentences with BM25 in the knowledge store."
    )
    parser.add_argument(
        "-d",
        '--data_path',
        type=str, default="data",
        help="The path of the data directory.",
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        type=str,
        default="sentences",
        help="sentences/qa",
    )
    parser.add_argument(
        "-p",
        "--pred_filename_suffix",
        type=str,
        default="top200.sent.json",
        help="The path of the file that stores predicted top-100 sentences for each claim.",
    )

    args = parser.parse_args()

    # for split in ["debug", "dev", "train", "test"]:
    for split in ["dev", "train"]:

        with open("{0}/{1}.json".format(args.data_path, split), "r", encoding="utf-8") as json_file:
            gold_data = json.load(json_file)

        input_path = "{0}/{1}{2}".format(args.data_path, split, args.pred_filename_suffix)
        output_path = "{0}/{1}.pos_neg.{2}.jsonl".format(args.data_path, split, args.data_type)

        with open(input_path, "r", encoding="utf-8") as in_file, open(output_path, "w") as out_file:
            for i, line in enumerate(in_file):
                pred_line = json.loads(line)
                gold_line = gold_data[i]
                assert pred_line['claim'] == gold_line['claim'], print(pred_line["claim"] +"\n"+ gold_line["claim"])
                line = generate_lines(pred_line, gold_line, split)
                out_file.write(json.dumps(line) + "\n")

