# src/main.py (PyTorch entry mimicking original project structure)
import os
import argparse
import time
import torch
import numpy as np
from types import SimpleNamespace

from data import Data, DataPlus
from model import Learner
from experiment import Experiment

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))

def main():
    parser = argparse.ArgumentParser(description="DRUM PyTorch Experiment")
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--rule_thr', default=1e-2, type=float)
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False, action="store_true")
    parser.add_argument('--exps_dir', default="exps", type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--datadir', default=None, type=str)
    parser.add_argument('--resplit', default=False, action="store_true")
    parser.add_argument('--no_link_percent', default=0., type=float)
    parser.add_argument('--type_check', default=False, action="store_true")
    parser.add_argument('--domain_size', default=128, type=int)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    parser.add_argument('--query_is_language', default=False, action="store_true")
    parser.add_argument('--vocab_embed_size', default=128, type=int)
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--rank', default=3, type=int)
    parser.add_argument('--rnn_state_size', default=128, type=int)
    parser.add_argument('--query_embed_size', default=128, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--get_phead', default=False, action="store_true")
    parser.add_argument('--adv_rank', default=False, action="store_true")
    parser.add_argument('--rand_break', default=False, action="store_true")
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)

    parser.add_argument('--sparse_to_dense_threshold', default=4096, type=int)
    parser.add_argument('--sparse_density_cutoff', default=0.02, type=float)

    args = parser.parse_args()
    d = vars(args)
    option = Option(d)
    if option.exp_name is None:
        option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
        option.tag = option.exp_name
    if option.resplit:
        assert not option.no_extra_facts
    if option.accuracy:
        assert option.top_k == 1

    if option.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not option.query_is_language:
        data = Data(option.datadir, option.seed, option.type_check, option.domain_size, option.no_extra_facts)
    else:
        data = DataPlus(option.datadir, option.seed)
    print("Data prepared.")

    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    if not option.query_is_language:
        option.num_query = data.num_query
    else:
        option.num_vocab = data.num_vocab
        option.num_word = data.num_word

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")
    option.sparse_to_dense_threshold = args.sparse_to_dense_threshold
    option.sparse_density_cutoff = args.sparse_density_cutoff

    option.save()
    print("Option saved.")

    learner = Learner(option.__dict__, device=device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=option.learning_rate)
    learner.set_optimizer(optimizer)
    print("Learner built.")

    if option.from_model_ckpt is not None and os.path.isfile(option.from_model_ckpt):
        learner.load_state_dict(torch.load(option.from_model_ckpt, map_location=device))
        print("Checkpoint restored from model %s" % option.from_model_ckpt)

    data.reset(option.batch_size)
    experiment = Experiment(option.__dict__, learner, data, device=device)
    print("Experiment created.")

    if not option.no_train:
        print("Start training...")
        experiment.train()

    if not option.no_preds:
        print("Start getting test predictions...")
        experiment.get_predictions()

    if option.get_vocab_embed:
        print("Start getting vocabulary embedding...")
        experiment.get_vocab_embedding()

    experiment.close_log_file()
    print("=" * 36 + "Finish" + "=" * 36)

if __name__ == "__main__":
    main()
