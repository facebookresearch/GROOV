# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from collections import defaultdict
import mwparserfromhell


"""
Utilities to parse XMC data into more convenient json format.

Example usage:

LABELS_PATH = YOUR_BASE_DIR/raw/AmazonCat-13K.raw/Yf.txt
INPUT_PATH = YOUR_BASE_DIR/raw/AmazonCat-13K.raw/trn.json
OUTPUT_PATH = YOUR_BASE_DIR/AmazonCat_13k_train.json

labels = load_labels(LABELS_PATH)
parse_data(INPUT_PATH, OUTPUT_PATH, labels, wiki_data=False)
"""


def load_labels(input_path):
    labels = []
    try:
        with open(input_path, "r") as ip_fp:
            for line in ip_fp:
                labels.append(line)
    except UnicodeDecodeError:
        with open(input_path, "r", encoding="ISO-8859-1") as ip_fp:
            for line in ip_fp:
                labels.append(line)
            
    return labels
        
def clean_tag(ip_tag):
    op_tag = ip_tag.split("->")[1]
    op_tag = op_tag.strip()
    op_tag = op_tag.replace('_', ' ')
    return op_tag.strip()

def parse_data(ip_json_path, op_json_path, labels, wiki_data=False, to_print=10):
    cnt = 0
    with open(ip_json_path, "r") as ip_fp, open(op_json_path, "w") as op_fp:
        for line in ip_fp:
            if cnt % 100000 == 0:
                print(f"{cnt} lines loaded!")
            op_ex = defaultdict()
            ip_ex = json.loads(line)
            
            op_ex['uid'] = ip_ex['uid']
            if wiki_data:
                op_ex['input'] = ip_ex['title'].replace("_", " ").strip() + " " + mwparserfromhell.parse(ip_ex['content']).strip_code().strip()
                op_ex['output'] = [clean_tag(labels[tag_idx]) for tag_idx in ip_ex['target_ind']]
            else:
                op_ex['input'] = ip_ex['title'].strip() + " " + ip_ex['content'].strip()
                op_ex['output'] = [labels[tag_idx].strip() for tag_idx in ip_ex['target_ind']]
                        
            if cnt < to_print:
                print("===============================================")
                print(f"UID: {op_ex['uid']}")
                print(f"Input: {op_ex['input'][:2000]}")
                print(f"Output: {op_ex['output']}")
            op_fp.write(json.dumps(op_ex) + "\n")
            cnt += 1