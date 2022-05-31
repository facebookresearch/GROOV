# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict
import functools
import json
import math
import os
import random
from time import time, strftime
import torch
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm

from torch.utils.data import DataLoader, SequentialSampler

from data_utils import Seq2SetDataset
from decode_utils import score_labels_by_probability_sum, LabelTrie
from local_configs import LOCAL_DATA_DIR
from s2s_model import make_s2s_batch, compute_metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import prepare_tokenizer
from params import ArgumentsS2S

def show_oov_samples(preds, golds, seen_label_set):
    preds_at_pos = defaultdict(int)
    oov_samples_by_pos = defaultdict(list)
    for p, g in zip(preds, golds):
        for pos, label in enumerate(p):
            preds_at_pos[pos] += 1
            if label not in seen_label_set:
                oov_samples_by_pos[pos].append((label, p, g))

    for pos in range(5):
        oov_rate = len(oov_samples_by_pos[pos]) * 1.0 / preds_at_pos[pos] if preds_at_pos[pos] > 0 else 0
        print(
            f"\n *** OOV rate at position {pos}: {oov_rate} ***"
        )
        if len(oov_samples_by_pos[pos]) >= 5:
            for oov_pred, prediction, gold_labels in random.sample(
                oov_samples_by_pos[pos], 5
            ):
                print("\n  OOV prediction: ", oov_pred)
                print("  Predicted: ", prediction)
                print("  Gold: ", gold_labels)


def decode_s2s(model, dataset, label_set, tokenizer, args, sep_token):
    model.eval()
    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_s2s_batch,
        model=model,
        tokenizer=tokenizer,
        max_i_len=args.max_i_length,
        max_o_len=args.max_o_length,
        device=args.device,
        add_example_ids=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        sampler=train_sampler,
        collate_fn=model_collate_fn,
    )
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()

    if args.decode_on_lattice:
        # Build trie of all possible labels
        label_trie = LabelTrie.from_labels(label_set, tokenizer, sep_token)

    def decode_on_label_lattice(batch_id, input_ids):
        next_tokens, completed_label_ids = label_trie.next_allowed_token(
            input_ids.tolist()[1:]
        )

        # Uncomment this to debug the decoding process
        # if batch_id == 0:
        #     input_tokens = tokenizer.decode(input_ids[1:], skip_special_tokens=False)
        #     print(" " * len (input_ids), input_ids.tolist()[1:], f'"{input_tokens}"', next_tokens if len(next_tokens) < 100 else "all tokens")

        return next_tokens

    with torch.no_grad():
        preds_by_method = defaultdict(list)
        golds = []
        for step, batch_inputs in enumerate(epoch_iterator):

            example_ids = batch_inputs["example_ids"]
            del batch_inputs["example_ids"]

            pre_loss = model(**batch_inputs)[0]

            if isinstance(model, DataParallel):
                model_gen = model.module
                loss = pre_loss.sum() / pre_loss.shape[0]
            else:
                model_gen = model
                loss = pre_loss

            generated_ids = model_gen.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                min_length=1,
                max_length=args.max_o_length + 1,
                do_sample=False,
                early_stopping=True,
                num_beams=args.decode_beams,
                temperature=1.0,
                top_k=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_return_sequences=args.decode_beams,
                decoder_start_token_id=tokenizer.bos_token_id,
                prefix_allowed_tokens_fn=decode_on_label_lattice
                if args.decode_on_lattice
                else None,
                return_dict_in_generate=True,
                output_scores=True,
            )

            if args.decode_beams > 1:
                # Use beam search to find most likely sequences and integrate over those sequences

                # Decoder doesn't have a separate dimension for beam size, need to reshape.
                # num_examples might be less than args.eval_batch_size in the last batch
                num_examples = generated_ids["sequences"].size()[0] // args.decode_beams
                for example_id, sequences, scores, labels in zip(
                    example_ids,
                    generated_ids["sequences"].view(
                        num_examples, args.decode_beams, -1
                    ),
                    generated_ids["sequences_scores"].view(
                        num_examples, args.decode_beams, 1
                    ),
                    batch_inputs["labels"],
                ):
                    top_label_sequences = []
                    top_scores = []
                    for sequence in sequences:
                        top_label_sequences.append(
                            dataset.token_ids_to_labels(tokenizer, sequence)
                        )

                    naive_preds = top_label_sequences[0]
                    filtered_preds = [l for l in naive_preds if l in label_set]
                    sum_prob_preds = score_labels_by_probability_sum(
                        top_label_sequences,
                        scores,
                    )
                    filtered_sum_prob_preds = [
                        l for l in sum_prob_preds if l in label_set
                    ]

                    preds_by_method["naive"].append((example_id, naive_preds))
                    preds_by_method["filtered"].append((example_id, filtered_preds))
                    preds_by_method["sum_prob"].append((example_id, sum_prob_preds))
                    preds_by_method["filtered_sum_prob"].append((example_id, filtered_sum_prob_preds))
                    golds.append(dataset.token_ids_to_labels(tokenizer, labels))
            else:
                # No beam search, work with the simple greedy output sequence
                for example_id, output_token_ids, label_token_ids in zip(
                    example_ids, generated_ids["sequences"], batch_inputs["labels"]
                ):

                    naive_preds = dataset.token_ids_to_labels(
                        tokenizer, output_token_ids
                    )
                    filtered_preds = [l for l in naive_preds if l in label_set]
                    gold = dataset.token_ids_to_labels(tokenizer, label_token_ids)

                    preds_by_method["naive"].append((example_id, naive_preds))
                    preds_by_method["filtered"].append((example_id, filtered_preds))
                    golds.append(gold)

            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    "{:5d} of {:5d}".format(step, len(dataset) // args.eval_batch_size)
                )

            # For the impatient kind
            # if len(golds) % 100 == 0:
            #     for method, preds in preds_by_method.items():
            #         metrics = compute_metrics(preds, golds)
            #         print(f"  {method}: " + " ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))

    print("Loss: {:.3f}".format(loc_loss / loc_steps))

    for method, preds in preds_by_method.items():

        metrics = compute_metrics([x[1] for x in preds], golds)
        print(f"{method}: " + " ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        method_str = method + ("_lattice" if args.decode_on_lattice else "")
        preds_file = os.path.join(LOCAL_DATA_DIR, args.output_dir, f"{args.pred_file_prefix}_{method_str}.jsonl")
        with open(preds_file, "w") as outfile:
            for id, preds in preds:
                outfile.write(
                    json.dumps(
                        {
                            "id": id,
                            "output": preds
                        }
                    ) + "\n"
                )

    golds_file = os.path.join(LOCAL_DATA_DIR, args.output_dir, f"{args.pred_file_prefix}_golds.jsonl")
    with open(golds_file, "w") as outfile:
        json.dump(golds, outfile)

    # Show stats and examples for the raw model output
    preds = [x[1] for x in preds_by_method["naive"]]

    show_oov_samples(preds, golds, label_set)


parser = ArgumentsS2S(decode_mode=True)
args = parser.parse_args()

model = AutoModelForSeq2SeqLM.from_pretrained(
    os.path.join(LOCAL_DATA_DIR, args.output_dir)
).to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
prepare_tokenizer(tokenizer)
sep = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"
print("sep token: ", sep)

s2s_train_set = Seq2SetDataset(
    os.path.join(LOCAL_DATA_DIR, args.train_file_path),
    sep,
    replace_underscores=args.replace_underscores,
)
s2s_dev_set = Seq2SetDataset(
    os.path.join(LOCAL_DATA_DIR, args.test_file_path),
    sep,
    replace_underscores=args.replace_underscores,
)

s2s_train_set.read_data()
s2s_dev_set.read_data()

train_label_set = s2s_train_set.get_all_labels()
dev_label_set = s2s_dev_set.get_all_labels()
print("# of distinct labels in train set:", len(train_label_set))
print("# of distinct labels in dev set:", len(dev_label_set))
print("# of new labels in dev set:", len(dev_label_set.difference(train_label_set)))
all_labels_set = train_label_set.union(dev_label_set)


decode_s2s(model, s2s_dev_set, all_labels_set, tokenizer, args, sep)
