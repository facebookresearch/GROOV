# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil

import torch

from data_utils import Seq2SetDataset
from decode_utils import LabelTrie
from local_configs import LOCAL_DATA_DIR
from params import ArgumentsS2S
from s2s_model import make_s2s_model, train_s2s
from utils import prepare_tokenizer

parser = ArgumentsS2S()
s2s_args = parser.parse_args()

# Prepare directory where we store output
if not os.path.isdir(os.path.join(LOCAL_DATA_DIR, "experiments")):
    os.mkdir(os.path.join(LOCAL_DATA_DIR, "experiments"))
output_dir = os.path.join(LOCAL_DATA_DIR, "experiments", s2s_args.output_dir)

if s2s_args.output_dir == "tmp":
    # This is the default value, we will just overwrite files here. Do not store important stuff in tmp!
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
else:
    # If we specified a directory, we want to make sure we don't accidentally overwrite something
    assert not os.path.isdir(output_dir), "Output directory already exists!"

os.mkdir(output_dir)

args_str = '\n'.join(f'{k} : {v}' for k, v in vars(s2s_args).items())
print("*** ARGS ***\n", args_str, "\n******")
with open(os.path.join(output_dir, "args.txt"), "w") as f:
    f.write(args_str)

# Initialize model and tokenizer
s2s_scheduler, s2s_optimizer, tokenizer, model, best_eval = make_s2s_model(
    model_name=s2s_args.model_name_or_path, from_file=None, device=s2s_args.device
)

# Construct trie used for decoding and multi-option loss
prepare_tokenizer(tokenizer)
sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

# Prepare datasets
train_data = os.path.join(LOCAL_DATA_DIR, s2s_args.train_file_path)
test_data = os.path.join(LOCAL_DATA_DIR, s2s_args.test_file_path)
s2s_train_set = Seq2SetDataset(
    train_data, sep_token, replace_underscores=s2s_args.replace_underscores, single_label=s2s_args.single_label
)
s2s_dev_set = Seq2SetDataset(
    test_data, sep_token, replace_underscores=s2s_args.replace_underscores, single_label=s2s_args.single_label
)

s2s_train_set.read_data()
s2s_dev_set.read_data()

# Amazon dataset has a particular label that sometimes causes the same label to appear
# repeatedly in the output after tokenization, breaking the set assumption and thus the code.
print("Sanity checking data...")
s2s_train_set.dedupe_data(tokenizer)
s2s_dev_set.dedupe_data(tokenizer)
print("Done!")


if s2s_args.use_multisoftmax:
    # To allow for any possible next token at a given time, we need a label trie
    # that will compute the corresponding target tensors.
    print("Computing label trie...")
    label_trie = LabelTrie.from_labels(
        s2s_train_set.get_all_labels().union(s2s_dev_set.get_all_labels()),
        tokenizer,
        sep_token,
    )
    print("Done!")
else:
    label_trie = None

# for using on multiple gpu using DataParallel
if s2s_args.data_parallel:
    s2s_model = torch.nn.DataParallel(model)
else:
    s2s_model = model

train_s2s(
    s2s_model,
    tokenizer,
    label_trie,
    s2s_optimizer,
    s2s_scheduler,
    best_eval,
    s2s_train_set,
    s2s_dev_set,
    s2s_args,
    output_dir
)
