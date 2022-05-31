# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Compute Propensity-Scored Precision @ K metric for a given dataset and prediction file.

Related paper: http://manikvarma.org/pubs/jain16.pdf

Sample commands:

python3 eval_psp.py \
    --train /private/home/danielsimig/ECG/data/Eurlex_4.3K/trn.jsonl \
    --gold /private/home/danielsimig/ECG/data/Eurlex_4.3K/tst.jsonl \
    --guess /private/home/danielsimig/ECG/data/experiments/eurlex_msm_t5-base_b32_48429737/epoch39/test_preds_sum_prob.jsonl

python3 eval_psp.py --A 0.5 --B 0.4\
    --train /private/home/danielsimig/ECG/data/Wikipedia_1M/trn.jsonl \
    --gold /private/home/danielsimig/ECG/data/Wikipedia_1M/sample_test.jsonl \
    --guess /private/home/danielsimig/ECG/data/experiments/wikipedia_1m_van_t5-base_b32_48432070/epoch0/test_preds_sum_prob.jsonl

"""


import argparse
import numpy as np
import json
import scipy.sparse as sp

# You'll need to install https://github.com/kunaldahiya/pyxclib for this
from xclib.evaluation.xc_metrics import Metrics, compute_inv_propesity, psprecision

def normalize_label(s):
    def remove_underscore(text):
        return text.replace("_", " ")

    def lower(text):
        return text.lower()

    return remove_underscore(lower(s))


def load_label_map(train_file, test_file):
    # Assign an index to every label so that we can build a sparse representation later
    label_idx_map = {}

    print("scanning train file")
    with open(train_file) as f:
        for line in f:
            for label in json.loads(line[:-1])['output']:
                if normalize_label(label) not in label_idx_map:
                    label_idx_map[normalize_label(label)] = len(label_idx_map)

    print("scanning test file")                
    with open(test_file) as f:
        for line in f:
            for label in json.loads(line[:-1])['output']:
                if normalize_label(label) not in label_idx_map:
                    label_idx_map[normalize_label(label)] = len(label_idx_map)

    return label_idx_map


def load_file(path, label_idx_map, preserve_order=False):
    
    # By default just load binary matrices. For preds use anything that preserves order
    score_fn = lambda x: 1
    if preserve_order:
        score_fn = lambda x: 1000-x
    
    unseen_labels = set()
    with open(path) as f:
        rows = []
        for line in f:
            row = json.loads(line[:-1])
            rows.append(row['output'])
            
    data = []
    ids = []
    label_ids = []
    
    for rid, row in enumerate(rows):
        for rank, label in enumerate(row):
            if normalize_label(label) not in label_idx_map:
                unseen_labels.add(normalize_label(label))
                continue
            data.append(score_fn(rank))
            ids.append(rid)
            label_ids.append(label_idx_map[normalize_label(label)])
    
    Y = sp.csc_matrix((data, (ids, label_ids)), shape = (len(rows), len(label_idx_map)))
            
    if len(unseen_labels) > 0:
        print(path, "contains", len(unseen_labels), "unseen labels")
    return Y



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--guess", help="Model prediction filepath")
    parser.add_argument("--gold", help="Gold labels filepath")
    parser.add_argument("--train", help="Training data filepath")
    parser.add_argument(
        "--A",
        type=float,
        default=0.55,
        help="Parameter A for calculating propensity. Should be 0.5 for Wiki, 0.55 otherwise.",
    )
    parser.add_argument(
        "--B",
        type=float,
        default=1.5,
        help="Parameter B for calculating propensity. Should be 0.4 for Wiki, 1.5 otherwise",
    )
    parser.add_argument(
        "--ks",
        type=str,
        required=False,
        default="1,3,5,10",
        help="Comma separated list of positive integers for recall@k and precision@k",
    )

    args = parser.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    label_idx_map = load_label_map(args.train, args.gold)

    print("Loading train file...")
    train_Y = load_file(args.train, label_idx_map)
    print("Shape: ", train_Y.shape)

    print("loading gold file")
    gold_Y = load_file(args.gold, label_idx_map)
    print("Shape: ", gold_Y.shape)

    print("loading guess file")
    guess_Y = load_file(args.guess, label_idx_map, preserve_order=True)
    print("Shape: ", guess_Y.shape)

    inv_label_propensities = compute_inv_propesity(train_Y, A=0.5, B = 0.4)
    psp_at_k = psprecision(guess_Y, gold_Y, inv_label_propensities, k=max(ks))
    for k in ks:
        print(f"PSP@{k}: {psp_at_k[k-1]:.3f}")