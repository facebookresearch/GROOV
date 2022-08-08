# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import json
import xmltodict
import xml.etree.ElementTree as ET
from pathlib import Path


def save_data(filename, data, encoding = 'utf8'):
    with open(filename, 'w', encoding=encoding) as fout:
        for res in data:
            json.dump(res, fout, ensure_ascii=False)
            fout.write("\n")

def parse_wned_dataset(dataset_folder):
    """Convert WNED dataset to entity tagging format
        - dataset_folder: path to WNED dataset
        return: List[dict]
    """
    
    dataset_name = dataset_folder.name
    xml_filepath = dataset_folder / f"{dataset_name}.xml"
    rawtext_folder = dataset_folder / "RawText"

    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    processed_data = []

    for document in root.findall('document'):
        doc_name = document.get('docName')
        doc_id = dataset_name + '-' + doc_name

        with open(rawtext_folder / doc_name, 'r') as fin:
            # Extract input context
            input_context = fin.read()

            # Extract entity annotations
            entity_set = set()
            for annotation in document.findall('annotation'):
                entity = annotation.find('wikiName').text
                if entity is not None and entity != 'NIL':
                    # print(entity)
                    entity_set.add(entity)

        processed_data.append({'id': doc_id, 'input': input_context, 'output': list(entity_set)})

    return processed_data


if __name__ == "__main__":
    datasets_folder = "/eval_datasets/basic_data/test_datasets/wned-datasets"
    for dataset_path in glob.glob(f'{datasets_folder}/*/'):
        print("Currently processing, ", dataset_path)
        
        dataset_path = Path(dataset_path)
        et_data = parse_wned_dataset(dataset_path)
        
        output_path = "processed_data"
        os.makedirs(output_path, exist_ok=True)
        output_file = output_path / f"{dataset_path.name}.jsonl"
        
        save_data(output_file, et_data)
