# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import numpy as np
import pprint
from collections import defaultdict, OrderedDict


def load_data(filename):
    data = []
    if ".jsonl" in filename:
        with open(filename, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                data.append(json.loads(line))
    else:
        with open(filename, "r") as fin:
            data = json.load(fin)
    return data


def normalize_label(s):
    def remove_underscore(text):
        return text.replace("_", " ")

    def lower(text):
        return text.lower()

    return remove_underscore(lower(s))


def get_rank(guess_item, gold_item):

    ground_truth = set()

    for label in gold_item["output"]:
        ground_truth.add(normalize_label(label))

    rank = []
    for label in guess_item["output"]:
        if normalize_label(label) in ground_truth:
            rank.append(True)
        else:
            rank.append(False)

    return rank, len(ground_truth)


# 1. Precision computation
def _precision(rank):
    if len(rank) == 0:
        return 0

    p = rank.count(True) / len(rank)

    return p

def _precision_at_k(rank, k):

    # precision @ k
    p = rank[:k].count(True) / k

    return p

def _propensity_scored_precision_at_k(rank, guess_inv_propensity_scores, gold_inv_propensity_scores, k):

    # Sum of inverse propensities for correct labels in top K
    num = sum(ps for correct, ps in zip(rank[:k], guess_inv_propensity_scores[:k]) if correct)
    # The maximum achievable propensity sum in top K
    den = sum(gold_inv_propensity_scores[:k])

    return num / den


# 2. Recall computation
def _recall(rank, num_distinct_labels):
    if num_distinct_labels == 0:
        return 0

    r = rank.count(True) / num_distinct_labels

    return r

def _recall_at_k(rank, num_distinct_labels, k):

    r = rank[:k].count(True) / num_distinct_labels

    return r


# 3. F1 computation
def _f1(rank, num_distinct_labels):
    
    p = _precision(rank)
    r = _recall(rank, num_distinct_labels)

    try:
        f1 = (2 * p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0

    return f1

def _f1_at_k(rank, num_distinct_labels, k):

    p = _precision_at_k(rank, k)
    r = _recall_at_k(rank, num_distinct_labels, k)
    
    try:
        f1 = (2 * p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0

    return f1


def get_ranking_metrics(guess_item, gold_item, ks, inv_propensity_scores_dict=None):

    P_at_k = {"precision@{}".format(k): 0 for k in sorted(ks) if k > 0}
    PSP_at_k = {"PSP@{}".format(k): 0 for k in sorted(ks) if k > 0}
    R_at_k = {"recall@{}".format(k): 0 for k in sorted(ks) if k > 1}
    F1_at_k = {"f1@{}".format(k): 0 for k in sorted(ks) if k > 1}

    assert (
        "output" in guess_item
    ), "guess should provide the output for {}".format(guess_item['id'])

    for k in ks:

        # 0. get rank
        rank, num_distinct_labels = get_rank(guess_item, gold_item)

        if inv_propensity_scores_dict is not None:
            # The less frequent the predicted label is the more it adds to the score
            guess_inv_propensity_scores = [inv_propensity_scores_dict[normalize_label(l)] for l in guess_item['output']]

            # Top propensity scores are used to compute the denominator
            gold_inv_propensity_scores = sorted(
                [inv_propensity_scores_dict[normalize_label(l)] for l in gold_item['output']],
                key=lambda x: -x
            )


        if num_distinct_labels > 0:

            # 1. precision
            P_at_k["precision@{}".format(k)] = _precision_at_k(rank, k)

            if inv_propensity_scores_dict is not None:
                PSP_at_k["PSP@{}".format(k)] = _propensity_scored_precision_at_k(
                    rank,
                    guess_inv_propensity_scores,
                    gold_inv_propensity_scores,
                    k,
                )

            # 2. recall
            R_at_k["recall@{}".format(k)] = _recall_at_k(rank, num_distinct_labels, k)

            # 3. F1 score
            F1_at_k["f1@{}".format(k)] = _f1_at_k(rank, num_distinct_labels, k)

    if inv_propensity_scores_dict is not None:
        return {**P_at_k, **PSP_at_k, **R_at_k, **F1_at_k}
    else:
        return {**P_at_k, **R_at_k, **F1_at_k}

def compute(gold_dataset, guess_dataset, ks=None, inv_propensity_scores_dict=None):

    result = {"precision": 0, "recall": 0, "f1": 0}
    if ks:
        ks = sorted([int(x) for x in ks])
        for k in ks:
            if k > 0:
                result["precision@{}".format(k)] = 0.0
                if inv_propensity_scores_dict is not None:
                    result["PSP@{}".format(k)] = 0.0

            if k > 1:
                result["recall@{}".format(k)] = 0.0
                result["f1@{}".format(k)] = 0.0

    assert len(guess_dataset) == len(
        gold_dataset
    ), "different size gold: {} guess: {}".format(len(guess_dataset), len(gold_dataset))

    for guess, gold in zip(guess_dataset, gold_dataset):
        id_key = "id" if "id" in gold else "uid"
        try:
            assert (
                str(gold[id_key]).strip() == str(guess["id"]).strip()
            ), "Items must have same order with same IDs"
        except KeyError:
            print(gold)
            print(guess)
            raise Exception
    
    for guess_item, gold_item in zip(guess_dataset, gold_dataset):
        
        # Aggregate rank-independent metrics
        rank, num_distinct_labels = get_rank(guess_item, gold_item)
        result["precision"] += _precision(rank)
        result["recall"] += _recall(rank, num_distinct_labels)
        result["f1"] += _f1(rank, num_distinct_labels)
        
        # Aggregate rank-based metrics
        if ks:
            ranking_metrics = get_ranking_metrics(
                guess_item, gold_item, ks, inv_propensity_scores_dict=inv_propensity_scores_dict
            )
            for k in ks:
                if k > 0:
                    result["precision@{}".format(k)] += ranking_metrics[
                        "precision@{}".format(k)
                    ]
                    if inv_propensity_scores_dict is not None:
                        result["PSP@{}".format(k)] += ranking_metrics[
                            "PSP@{}".format(k)
                        ]

                if k > 1:
                    result["recall@{}".format(k)] += ranking_metrics["recall@{}".format(k)]
                    result["f1@{}".format(k)] += ranking_metrics["f1@{}".format(k)]

    if len(guess_dataset) > 0:
        result["precision"] /= len(guess_dataset)
        result["recall"] /= len(guess_dataset)
        result["f1"] /= len(guess_dataset)
        
        if ks:
            for k in ks:
                if k > 0:
                    result["precision@{}".format(k)] /= len(guess_dataset)
                    if inv_propensity_scores_dict is not None:
                        result["PSP@{}".format(k)] /= len(guess_dataset)
                if k > 1:
                    result["recall@{}".format(k)] /= len(guess_dataset)
                    result["f1@{}".format(k)] /= len(guess_dataset)

    return OrderedDict(sorted(result.items(), key=lambda x: x[0]))

def inv_propensity_formula(label_frequency, num_instances, A=0.55, B=1.5):
    # Code based on: https://fburl.com/rp6rqhvg
    # Related paper: http://manikvarma.org/pubs/jain16.pdf

    C = (np.log(num_instances)-1)*np.power(B+1, A)
    return 1.0 + C*np.power(label_frequency+B, -A) 


def compute_inv_label_propensities(label_frequencies, num_instances):
    return defaultdict(
        lambda: inv_propensity_formula(0, num_instances),
        {normalize_label(label): inv_propensity_formula(freq, num_instances) for label, freq in label_frequencies.items()}
    )


def evaluate(gold, guess, ks=None, freqs=None):
    pp = pprint.PrettyPrinter(indent=4)

    gold_records = load_data(gold)
    guess_records = load_data(guess)

    assert len(gold_records) == len(guess_records)

    if freqs is not None:
        with open(freqs) as f:
            label_frequencies = json.load(f)
        inv_propensity_scores_dict = compute_inv_label_propensities(label_frequencies, 450000)  #TODO
    else:
        inv_propensity_scores_dict = None


    # 2. get retrieval metrics
    result = compute(gold_records, guess_records, ks, inv_propensity_scores_dict)

    pp.pprint(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--guess", help="Guess KILT file")
    parser.add_argument("--gold", help="Gold KILT file")

    parser.add_argument(
        "--ks",
        type=str,
        required=False,
        default=None,
        help="Comma separated list of positive integers for recall@k and precision@k",
    )

    parser.add_argument(
        "--freqs",
        type=str,
        required=False,
        default=None,
        help="JSON file containing frequencies of labels in training data. If this is specified, we'll compued propensity-weighted P@K metrics."
    )

    args = parser.parse_args()

    if args.ks:
        args.ks = [int(k) for k in args.ks.split(",") if int(k) > 0]

    evaluate(args.gold, args.guess, args.ks, args.freqs)
