# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from itertools import chain
from torch.utils.data import Dataset
import random


class Seq2SetDataset(Dataset):
    def __init__(self, path, sep, replace_underscores=False, 
                 read_per_line=False, single_label=False, output_key="output"):
        self.path = path
        self.sep = sep
        self.data = None
        self.label_set = None
        self.replace_underscores = replace_underscores
        self.read_per_line = read_per_line
        self.single_label = single_label

        self.output_key = output_key    # key for gold annotations 

    def read_data(self):

        print("Reading", self.path)

        with open(self.path, "r") as f:
            if self.path.split(".")[-1] == "jsonl":
                self.data = [json.loads(line) for line in f.readlines()]
            else:
                self.data = json.load(f)

        self.label_set = set(chain.from_iterable(row[self.output_key] for row in self.data))

    def dedupe_data(self, tokenizer):
        for i, line in enumerate(self.data):
            new_output = []
            tokenized_labels = []
            for label in line[self.output_key]:
                tokenized = tuple(tokenizer(label).input_ids[:-1])
                if tokenized not in tokenized_labels:
                    new_output.append(label)
                    tokenized_labels.append(tokenized)
                else:
                    print(f"Line {line['id']} has repeated labels!")
            line[self.output_key] = new_output
            if i % 10000 == 0:
                print(i)

    def __len__(self):
        assert (
            self.data is not None
        ), "Attempted to access data before loading it. Call read_data() first"
        return len(self.data)

    def order_labels(self, label_seq):
        # Comment this line to disable label shuffling
        random.shuffle(label_seq)
        return label_seq

    def label_to_str(self, label):
        return label.replace("_", " ") if self.replace_underscores else label

    def output_str_to_labels(self, output_str):
        return [x.strip() for x in output_str.split(self.sep)]

    def token_ids_to_labels(self, tokenizer, token_ids):
        return self.output_str_to_labels(
            tokenizer.decode(token_ids).split("</s>")[0].replace("<pad>", "")
        )

    def make_example(self, idx):

        assert (
            self.data is not None
        ), "Attempted to access data before loading it. Call read_data() first"

        example = self.data[idx]
        id = example["id"] if "id" in example else example["uid"]
        input = example["input"]
        labels = [random.choice(example[self.output_key])] if self.single_label else self.order_labels(example[self.output_key])

        out_str = self.sep.join(self.label_to_str(label) for label in labels)

        return (input.lower().strip(), out_str, id)

    def __getitem__(self, idx):
        return self.make_example(idx)

    def return_all_inputs(self):
        qs = []
        for i in range(len(self.data)):
            qs.append(self.make_example(i)[0])
        return qs

    def get_all_labels(self):
        assert (
            self.label_set is not None
        ), "Attempted to access data before loading it. Call read_data() first"
        return {self.label_to_str(label) for label in self.label_set}
