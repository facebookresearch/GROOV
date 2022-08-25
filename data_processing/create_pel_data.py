# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import json
import argparse
import os
from tqdm import tqdm
import random
import gzip
from transformers import AutoTokenizer


def load_data(filename, new_key=None, old_key=None):
    data = []
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            instance = json.loads(line)
            if new_key and old_key:    # Make sure instances in the merged dataset have the same output key for entities
                instance[new_key] = instance.pop(old_key)
            
            data.append(instance)

    return data

def save_data(filename, data, encoding = 'utf8'):
    with open(filename, 'w', encoding=encoding) as fout:
        for res in data:
            json.dump(res, fout, ensure_ascii=False)
            fout.write("\n")

# Remove tags with non-English characters, Wiktionary labels and 'None'
def dedupe_labels(labels, has_mention=False):
    processed_labels = []
    for label in labels:
        entity = label
        if has_mention:
            entity = label[-1]
            
        if not entity.isascii() or 'Wiktionary' in entity or entity == 'None':
            continue
            
        if label not in processed_labels:
            processed_labels.append(label)
            
    return processed_labels

# Create mention table from pre-computed annotation file
# https://github.com/masha-p/PPRforNED
# Convert the annotation to a dict object with format {"mention" : [ candidate entities, ..]}
def parse_annotation_file(file_path, mention_table):
    mention = None
    candidates = []
    with open(file_path, 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            parsed_row = line.split("\t")

            if parsed_row[0] == "ENTITY":
                if mention and len(candidates) > 0:
                    if mention not in mention_table:
                        mention_table[mention] = candidates

                mention = parsed_row[1].split(':')[1]
                candidates = []

            elif parsed_row[0] == 'CANDIDATE':
                url = parsed_row[5][4:]
                wikiname = url.rsplit('/', 1)[-1].replace('_', ' ')
                candidates.append(wikiname)

# Process mention annotations to parallel EL format
def remove_mention_text(mentions):
    for mention in mentions:
        del mention[2]

def yield_lines(filepath, n_lines=None):
    filepath = Path(filepath)
    with open(filepath, "rt") as f:
        for i, l in enumerate(f):
            if n_lines is not None and i >= n_lines:
                break
            yield l.rstrip("\n")

def yield_jsonl_lines(filepath, *args, **kwargs):
    for line in yield_lines(filepath, *args, **kwargs):
        yield json.loads(line)

def process_kilt_data(src_path, tar_path):
    with gzip.open(tar_path, "wt", compresslevel=1) as f:
        for data in yield_jsonl_lines(src_path):
            
            # Remove labels with special characters
            data['entities'] = dedupe_labels(data['entities'])

            # Remove duplicates in topic labels
            data['topics'] = dedupe_labels(data['topics'])
            
            # Skip illegal data with empty input or no labels
            if len(data['input']) == 0 or (len(data['entities']) == 0 or len(data['topics']) == 0):
                continue

            f.write(json.dumps(data) + "\n")

# Find the index of leftmost interval with desired property
def search_token_index(char_index, intervals):
    left, right = 0, len(intervals) - 1
    res = len(intervals)
    while left <= right:
        mid = (left + right) // 2
        start, end = intervals[mid][0], intervals[mid][1]
        if  start <= char_index <= end:
            res = min(res, mid)
            right = mid - 1
        elif char_index > end:
            left = mid + 1
        else:
            right = mid - 1
    
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        default="/data/KILT_entities_topics/KILT_entities_topics_train.jsonl",
        type=str,
        help="Path to Wikipedia abstract data"
    )

    parser.add_argument(
        "--tar_path",
        default="/data/KILT_entities_topics/pel_KILT_entities_topics_train.jsonl.gz",
        type=str,
        help="Path to processed data (in parallel EL format)"
    )

    parser.add_argument(
        "--mention_file",
        default="/data/mention_table.json",
        type=str,
        help="Path to pre-computed mention table"
    )

    parser.add_argument(
        "--entity_file",
        default="/data/entities.json",
        type=str,
        help="Path to file that contains candidate candidites"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    # Load pre-computed mention table
    with open(args.mention_file) as f:
        mention_table = json.load(f)

    # Load global entity set
    with open(args.entity_file) as f:
        entity_set = json.load(f)

    with gzip.open(args.tar_path, "wt", compresslevel=1) as f:
        for data in yield_jsonl_lines(args.src_path):
            data['anchors'] = list()
            encoded_input = tokenizer(data['input'], return_offsets_mapping=True)
            
            if len(encoded_input['input_ids']) >= 4096:
                print(data['id'])
                print(len(encoded_input['input_ids']))
                continue
            
            for mention in data['mentions']:
                # Convert character index to token index for each mention span
                char_start = mention['anchor']['start']
                char_end = mention['anchor']['end']
                token_start = search_token_index(char_start, encoded_input["offset_mapping"])
                token_end = search_token_index(char_end, encoded_input["offset_mapping"])
                
                data['anchors'].append([token_start, token_end, 
                                        mention['anchor']['text'], mention['entisty']])
            
            # Remove illegal or duplicate labels
            data['entities'] = dedupe_labels(data['entities'])
            data['topics'] = dedupe_labels(data['topics'])
            
            # Remove illegal labels in anchors (note that different mentions with same label will be kept)
            data['anchors'] = dedupe_labels(data['anchors'], has_mention=True)
            
            # Read candidate entities from pre-computed mention table
            # or generate one random label from global entity set if cannot find a match
            data['candidates'] = list()
            for mention in data['anchors']:
                mention_text = mention[2]
                if mention_text in mention_table:
                    data['candidates'].append(mention_table[mention_text])
                else:
                    data['candidates'].append([mention[-1], random.sample(entity_set, k=1)[0]])
            
            remove_mention_text(data['anchors'])
            
            # Skip illegal instance with empty input or no labels/mentions
            if len(data['input']) == 0 or (len(data['entities']) == 0 or len(data['topics']) == 0) or len(data['anchors']) == 0:
                continue
            
            data.pop('paragraphs')
            data.pop('mentions')
            
            f.write(json.dumps(data) + "\n")
