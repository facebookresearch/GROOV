# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from collections import defaultdict
import json
import os
import random

DATA_ROOT = "/checkpoint/danielsimig/ECG/data_test/"

# Data in the format of extract_data.py
ORIG_TRAIN_FILE = os.path.join(DATA_ROOT, "AmazonCat_13k_train.jsonl")
ORIG_TEST_FILE = os.path.join(DATA_ROOT, "AmazonCat_13k_test.jsonl")

# In this folder we're going to produce a set of files, with increasingly more data moved. 
# We prioduce versions with 1K, 2k, .. 10K moved labels. 1K is eventually used in the paper.
SHUFFLED_FOLDER = os.path.join(DATA_ROOT, "AmazonCat_OOV/")
REMOVE_STEP = 1000

# Don't move too frequent labels. It's unrealistic that those labels are novel and doing so
# would decrease our train set too much anyway.
MAX_FREQ_TO_REMOVE = 1000


def read_data(path):
    orig = {}
    labels = set()
    label_freqs = defaultdict(int)
    with open(path) as f:
        for line in f.readlines():
            line = json.loads(line)
            orig[line['uid']] = line
            for label in line['output']:
                label = label.replace("_", " ")
                labels.add(label)
                label_freqs[label] += 1

    return orig, labels, label_freqs


def write_data(data, path):
    with open(path, "w") as f:
        for line in data.values():
            f.write(json.dumps(line) + "\n")


train_orig, train_labels, train_label_freqs = read_data(ORIG_TRAIN_FILE)
test_orig, test_labels, test_label_freqs = read_data(ORIG_TEST_FILE)

random.seed(0)
remove_order = [k for k, v in train_label_freqs.items() if v < MAX_FREQ_TO_REMOVE]
random.shuffle(remove_order)

train_shuffled = copy.deepcopy(train_orig)
test_shuffled = copy.deepcopy(test_orig)

print(f"ORIGINAL SIZES:  Train: {len(train_shuffled)}  Test: {len(test_shuffled)}")

for i in range(10):
    removed_labels = set(remove_order[i*REMOVE_STEP: (i+1) * REMOVE_STEP])
    train_shuffled_tmp = {}
    num_oov_labels = 0
    for k, v in train_shuffled.items():
        # Move any data point with an occurance of a moved label to the test set
        if any(x in removed_labels for x in v['output']):
            test_shuffled[k] = v
        else:
            train_shuffled_tmp[k] = v
    train_shuffled = train_shuffled_tmp

    print(f"After moving {(i+1) * REMOVE_STEP} labels:\tTrain: {len(train_shuffled)} \tTest: {len(test_shuffled)}")

    write_data(train_shuffled, SHUFFLED_FOLDER + f"train_max1k_{(i+1)*REMOVE_STEP}_moved.jsonl")
    write_data(test_shuffled, SHUFFLED_FOLDER + f"test_max1k_{(i+1)*REMOVE_STEP}_moved.jsonl")

