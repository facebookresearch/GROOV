# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import math
import os
import random
from collections import defaultdict
from time import time, strftime

import torch
import torch.multiprocessing as mp
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, SubsetRandomSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    T5ForConditionalGeneration,
)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.nn import CrossEntropyLoss

from data_utils import Seq2SetDataset


def compute_metrics(preds, golds):
    metrics = defaultdict(float)

    num_datapoints = len(golds)
    assert len(preds) == num_datapoints

    for g, p in zip(golds, preds):
        g_labels = set(g)
        p_labels = set(p)
        inter = p_labels.intersection(g_labels)

        metrics["micro_accuracy"] += (
            (1.0) * len(inter) / len(p_labels) if len(p_labels) > 0 else 0.0
        )
        metrics["micro_recall"] += (1.0) * len(inter) / len(g_labels)
        for k in [1, 3, 5]:
            topk_inter = set(p[:k]).intersection(g_labels)
            metrics[f"P@{k}"] += (1.0) * len(topk_inter) / k

    return {k: v * 1.0 / num_datapoints for k, v in metrics.items()}


def make_s2s_model(model_name="facebook/bart-large", from_file=None, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if from_file is not None:
        # model = AutoModelForSeq2SeqLM.from_pretrained(from_file + "_dir").to(device)
        model = AutoModelForSeq2SeqLM.from_pretrained(from_file).to(device)
    elif "led-base" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, gradient_checkpointing=True, use_cache=False
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    s2s_optimizer = None
    s2s_scheduler = None
    best_eval = None
    start_epoch = None
    if from_file is not None:
        param_dict = torch.load(
            os.path.join(from_file, "state_dict.pth"), map_location=device
        )  # has model weights, optimizer, and scheduler states
        s2s_optimizer = AdamW(model.parameters(), lr=0.0001, eps=1e-8)
        s2s_scheduler = get_linear_schedule_with_warmup(
            s2s_optimizer,
            num_warmup_steps=400,
            num_training_steps=1,
        )
        s2s_optimizer.load_state_dict(param_dict["optimizer"])
        s2s_scheduler.load_state_dict(param_dict["scheduler"])
        if "loss_spearman" in param_dict["best_eval"]:
            best_eval = param_dict["best_eval"]["loss_spearman"]
        else:
            best_eval = param_dict["best_eval"]["loss"]

        if "epoch" in param_dict:
            start_epoch = int(param_dict["epoch"])

    return s2s_scheduler, s2s_optimizer, tokenizer, model, best_eval, start_epoch


def make_s2s_batch(
    io_list,
    tokenizer,
    model,
    label_trie=None,
    max_i_len=512,
    max_o_len=16,
    device="cuda:0",
    add_example_ids=False
):
    i_ls = [i for i, _, _ in io_list]
    o_ls = [o for _, o, _ in io_list]

    i_toks = tokenizer(
        i_ls, max_length=max_i_len, padding="max_length", truncation=True
    )
    i_ids, i_mask = (
        torch.LongTensor(i_toks["input_ids"]).to(device),
        torch.LongTensor(i_toks["attention_mask"]).to(device),
    )

    o_toks = tokenizer(
        o_ls, max_length=max_o_len + 1, padding="max_length", truncation=True
    )

    o_ids, o_mask = (
        torch.LongTensor(o_toks["input_ids"]).to(device),
        torch.LongTensor(o_toks["attention_mask"]).to(device),
    )

    # Based on HF examples
    if isinstance(model, DataParallel):
        model = model.module

    if isinstance(model, T5ForConditionalGeneration):
        decoder_input_ids = model._shift_right(o_ids)
        lm_labels = o_ids
    else:
        decoder_input_ids = o_ids[:, :-1].contiguous()
        lm_labels = o_ids[:, 1:].contiguous().clone()

    model_inputs = {
        "input_ids": i_ids,
        "attention_mask": i_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": lm_labels,
    }

    # Compute target for multi-option loss
    if label_trie:
        model_inputs["targets"] = label_trie.compute_targets(o_ids, o_ls)
    if add_example_ids:
        model_inputs["example_ids"] = [id for _, _, id in io_list]

    return model_inputs


def train_s2s_epoch(
    model,
    dataset,
    tokenizer,
    label_trie,
    optimizer,
    scheduler,
    args,
    output_dir,
    e=0,
    curriculum=False,
):
    model.train()
    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)

    tokenizer.source_len = [0.0, 0.0, 0.0, 0.0]
    model_collate_fn = functools.partial(
        make_s2s_batch,
        model=model,
        tokenizer=tokenizer,
        label_trie=label_trie,
        max_i_len=args.max_i_length,
        max_o_len=args.max_o_length,
        device=args.device,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=model_collate_fn,
    )
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)

    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch_inputs in enumerate(epoch_iterator):

        if args.use_multisoftmax:
            # Targets matrix that allows for all possible next tokens at a given time
            # Dimension: batch_size x max_output_len x num_tokens
            # Need to pass label_trie to model_collate_fn for this to work
            # Passing this as part of batch_inputs is a hack. The model doesn't actually expect this input so need to pop
            targets = batch_inputs.pop("targets", None).to(args.device)

        model_output = model(**batch_inputs)

        if args.use_multisoftmax:
            # Variaion on SoftMax that'll allow of to distribute weight over different outcomes

            # We'll compute the loss ourselves outside of the model. We need the raw logits for that.
            lm_logits = model_output[1]

            # We want to take the sum of the exp() of the logits correspodinging to possible next tokens.
            # For everything else we want 0, this is achieved by exp(-inf)
            gt_label_logits = lm_logits.masked_fill(targets == 0, float("-inf"))

            # Intuitively, this removes competition between correct labels
            # torch.logsumexp(gt_label_logits, dim=2) is just the multi-option equivalent of the
            # log(exp(gt_label_logit)) term in the single label case.
            multisoftmax_per_token = -torch.logsumexp(
                gt_label_logits, dim=2
            ) + torch.logsumexp(lm_logits, dim=2)

            pre_loss = torch.mean(multisoftmax_per_token)
        else:
            # Use the vanilla softmax of the HF model of choice
            pre_loss = model_output[0]

        loss = pre_loss
        loss.mean().backward()
        # optimizer
        if step % args.backward_freq == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        loc_loss += loss.mean().item()
        loc_steps += 1
        if step % args.print_freq == 0 or step == 1:
            print(
                "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    e,
                    step,
                    len(dataset) // args.train_batch_size,
                    loc_loss / loc_steps,
                    time() - st_time,
                )
            )
            with open(os.path.join(output_dir,  "output_train.txt"), "a+") as dev_file:
                dev_file.write(
                    "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}\n".format(
                        e,
                        step,
                        len(dataset) // args.train_batch_size,
                        loc_loss / loc_steps,
                        time() - st_time,
                    )
                )
            loc_loss = 0
            loc_steps = 0


def eval_s2s_epoch(model, dataset, tokenizer, args, output_dir, sample=None):
    model.eval()

    print("Eval with sampling rate", sample)

    num_examples = len(dataset)
    if sample is not None:
        num_examples_used = int(num_examples * sample)
        torch.manual_seed(0)
        eval_sampler = SubsetRandomSampler(indices=torch.randperm(num_examples)[:num_examples_used])
        num_examples = num_examples_used
    else:
        eval_sampler = SequentialSampler(dataset)

    model_collate_fn = functools.partial(
        make_s2s_batch,
        model=model,
        tokenizer=tokenizer,
        max_i_len=args.max_i_length,
        max_o_len=args.max_o_length,
        device=args.device,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=model_collate_fn,
    )
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()

    with torch.no_grad():
        preds = []
        golds = []
        for step, batch_inputs in enumerate(epoch_iterator):
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
                num_beams=1,
                temperature=1.0,
                top_k=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                decoder_start_token_id=tokenizer.bos_token_id,
            )

            generated_ids = list(generated_ids)

            raw_preds = [
                (tokenizer.decode(ans_ids).split("</s>")[0].replace("<pad>", ""))
                for ans_ids in generated_ids
            ]

            pred = [dataset.output_str_to_labels(pred) for pred in raw_preds]

            gold = [
                dataset.output_str_to_labels(
                    tokenizer.decode(ans_ids).split("</s>")[0].replace("<pad>", "")
                )
                for ans_ids in batch_inputs["labels"]
            ]

            # Print to quickly debug predictions
            # print(generated_ids[0])
            # print("raw_pred", raw_preds[0])
            # print("pred", pred[0])
            # print("gold", gold[0])

            golds.extend(gold)
            preds.extend(pred)

            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    "{:5d} of {:5d} \t L: {:.6f} \t  -- {:.3f}".format(
                        step,
                        num_examples // args.eval_batch_size,
                        loc_loss / loc_steps,
                        time() - st_time,
                    )
                )

    with open(os.path.join(output_dir, "predictions_dev.txt"), "a") as dev_file:
        for g, p in zip(golds, preds):
            dev_file.write(str(g) + "\t" + str(p) + "\n")

    metrics = compute_metrics(preds, golds)
    metric_str = "L: {:.3f} ".format(loc_loss / loc_steps) + " ".join(
        f"{k}: {v:.3f}" for k, v in metrics.items()
    )

    with open(os.path.join(output_dir, "output_dev.txt"), "a") as dev_file:
        dev_file.write(metric_str + "\n")
    print(metric_str)

    return loc_loss, metrics["P@3"]  # Use P@3 to decide best model


def save_checkpoint(output_dir, s2s_model, eval_acc, s2s_optimizer, s2s_scheduler, epoch):
    start_time = time()
    print("Saving checkpoint starts at", strftime('%l:%M%p %Z on %b %d, %Y'))

    best_eval = eval_acc
    m_save_dict = {
        "optimizer": s2s_optimizer.state_dict(),
        "scheduler": s2s_scheduler.state_dict(),
        "best_eval": {"em": eval_acc},
        "epoch": epoch
    }
    print("Saving model {}".format(output_dir))

    if isinstance(s2s_model, DataParallel):
        s2s_model.module.save_pretrained(output_dir)
    else:
        s2s_model.save_pretrained(output_dir)

    torch.save(m_save_dict, os.path.join(output_dir, "state_dict.pth"))
    print("Saving checkpoint took", int(time() - start_time), "seconds")


def train_s2s(
    s2s_model,
    s2s_tokenizer,
    label_trie,
    s2s_optimizer,
    s2s_scheduler,
    best_eval,
    s2s_train_dset,
    s2s_valid_dset,
    s2s_args,
    output_dir,
    start_epoch=0
):
    if s2s_optimizer is None:
        s2s_optimizer = AdamW(
            s2s_model.parameters(), lr=s2s_args.learning_rate, eps=1e-8
        )
    if s2s_scheduler is None:
        s2s_scheduler = get_linear_schedule_with_warmup(
            s2s_optimizer,
            num_warmup_steps=400,
            num_training_steps=(s2s_args.num_epochs + 1)
            * math.ceil(len(s2s_train_dset) / s2s_args.train_batch_size),
        )
    for e in range(start_epoch, start_epoch + s2s_args.num_epochs):
        train_s2s_epoch(
            s2s_model,
            s2s_train_dset,
            s2s_tokenizer,
            label_trie,
            s2s_optimizer,
            s2s_scheduler,
            s2s_args,
            output_dir,
            e,
            curriculum=(e == 0),
        )

        # Decoding can be slow, we can control how often we want to do that
        if e % s2s_args.eval_every_k_epoch == s2s_args.eval_every_k_epoch - 1:
            start_time = time()
            print("Eval starts at", strftime('%l:%M%p %Z on %b %d, %Y'))

            eval_l, eval_acc = eval_s2s_epoch(
                s2s_model,
                s2s_valid_dset,
                s2s_tokenizer,
                s2s_args,
                output_dir,
                sample=s2s_args.eval_sampling_rate
            )

            print("Evaluation took", int(time() - start_time), "seconds")

            if s2s_args.save_after_every_eval:
                checkpoint_dir = os.path.join(output_dir, f"epoch{e}")
                os.mkdir(checkpoint_dir)
                save_checkpoint(checkpoint_dir, s2s_model, eval_acc, s2s_optimizer, s2s_scheduler, e)
            elif best_eval == None or eval_acc > best_eval:
                save_checkpoint(output_dir, s2s_model, eval_acc, s2s_optimizer, s2s_scheduler, e)
