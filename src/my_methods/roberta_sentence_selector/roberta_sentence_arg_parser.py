import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        help="The input data dir. ", )
    parser.add_argument("--cache_dir", type=str,
                        help="cached data dir. ", )

    parser.add_argument("--ckpt_root_dir", default="bert_weights/roberta_sentence_selector_ckpt_root_dir",
                        type=str, help="The checkpoints dir. ", )
    parser.add_argument("--test_ckpt", default=None, type = str,
                        help = "The checkpoint name for test")
    parser.add_argument("--load_model_path", type=str,
                        help="Load model from the path.", )

    parser.add_argument("--model_type", default="RobertaCls", type=str,
                        help="Name of model.", )
    parser.add_argument("--bert_name", type=str,
                        help="name or path of pretrained bert.")
    parser.add_argument("--data_generator", default="RobertaSentenceGenerator", type=str,
                        help="Name of preprocessor.", )

    parser.add_argument("--force_generate", action="store_true",
                        help= "if set, generate data regardless of whether there is a cache")
    parser.add_argument("--save_all_ckpt", action="store_true")
    parser.add_argument("--test", action="store_true"
                        , help="whether to generate test results")

    parser.add_argument("--fix_bert", action="store_true",
                        help="whether to train bert parameters")

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)

    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--warm_rate", default=0.2, type=float)

    parser.add_argument("--seed", default=1234, type=int,
                        help="random seed")
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--max_epoch", default=3, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1, type=int)

    parser.add_argument('--wiki_path', type=str, default="data/feverous_wikiv1.db", help='/path/to/data')
    parser.add_argument("--predict_threshold", type=float, default=0.5)

    parser.add_argument('--max_sent', type=int, default=5)
    parser.add_argument('--retrieval_turns', type=int, default=3)

    parser.add_argument('--user_given_model_suffix', type=str, default='Default')
    parser.add_argument('--train_data_extend_multiplication', type=int, default=1)

    return parser
