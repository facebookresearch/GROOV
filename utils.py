# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import csv
import os
import pandas

from itertools import chain


def create_schema_from_train_test(train_labels, test_labels, schema_file):
    labels = {}
    counter = 0
    with open(train_labels) as f:
        lines = f.readlines()
        for line in lines:
            for l in line.split():
                if l not in labels:
                    labels[counter] = l
                    counter += 1
    with open(test_labels) as f:
        lines = f.readlines()
        for line in lines:
            for l in line.split():
                if l not in labels:
                    labels[counter] = l
                    counter += 1
    out_file = open(schema_file, "w")
    writer = csv.writer(out_file)
    for key, value in labels.items():
        writer.writerow([key, value])
    out_file.close()


# schema is given in a file, each line is ID, topic_name
def read_schema(data_file):
    with open(data_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        line_count = 0
        data = {}
        for row in reader:
            if line_count == 0:
                line_count = line_count + 1
                continue

            id = row[0]
            name = row[1]
            data[id] = name
    return data


def prepare_tokenizer(tokenizer):
    special_tokens = []
    special_tokens.extend(["<sep>", "<SEP>", "<eos>", "[SEP]"])
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})


def read_multilabel(data_file, labels):
    data_df = pandas.read_pickle(data_file)

    post_ids = data_df.post_id
    post_texts = data_df.post_text
    ocr_texts = data_df.ocr_text
    landing_texts = data_df.landing_page_text
    tags_list = data_df.tags
    data = []
    for post_id, post_text, ocr_text, landing_text, tags in zip(
        post_ids, post_texts, ocr_texts, landing_texts, tags_list
    ):
        tags_names = []
        for l in tags:
            if l in labels:
                tags_names.append(labels[l])
            else:
                print(l)
        data.append([post_id, post_text, ocr_text, landing_text, tags_names])
    return data


def try_convert(val):
    try:
        return float(val)
    except ValueError:
        return -1


if __name__ == "__main__":
    print("main")
    create_schema_from_train_test(
        "EUR-Lex/train_labels.txt", "EUR-Lex/test_labels.txt", "EUR-Lex/EUR-Lex-schema"
    )
