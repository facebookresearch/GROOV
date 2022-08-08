# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from collections import Counter
import json

import pandas as pd
import plotly.express as px


def load_data(filename):
    data = []
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))

    return data

def get_label_counter(dataset):
    label_list = []
    for instance in dataset:
        label_list.extend(instance['output'])
    label_counter = Counter(label_list)
    
    return label_counter

def create_data_frame(path: Path):
    """Create a DataFrame for AIDA data"""
    data_rows = []
    label_list = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            data_rows.append({
                "id": data['id'],
                "input": data['input'],
                "labels": data['output'],
                "num_labels": len(data['output']),
                "input_len": len(data['input'].split(' '))
            })
            label_list.extend(data['output'])

    df = pd.DataFrame(data_rows)
    label_counter = Counter(label_list)
    
    return df, label_counter

def visualize_data(path: Path):
    df, label_counter = create_data_frame(path)
    
    # Distribution of input length
    print(f"Total number of samples: {len(df)} \n")
    print('Input length: \n')
    fig = px.histogram(df.input_len, x="input_len")
    fig.show()
    
    # label distribution
    print('Label Distribution:\n')
    fig = px.histogram(df.num_labels, x="num_labels", nbins=20)
    fig.show()
    
    print('Most common labels: \n')
    label_df = pd.DataFrame(label_counter.most_common(), columns=["label", "count"])
    print(label_df.head)
    px.bar(label_df.head(50), x="label", y="count", title="Most common labels").show()

    print(f"Unique labels: {len(label_counter)}")
    print(f"Total labels: {sum(label_counter.values())}")
    
    # Outliers with large number of labels
    num_outliers_50 = len(df[df.num_labels > 50])
    num_outliers_100 = len(df[df.num_labels > 100])
    print('Number of outliers:\n')
    print(f"# of samples with 50+ labels: {num_outliers_50} ({num_outliers_50 / len(df) * 100:.4f} %)")
    print(f"# of samples with 100+ labels: {num_outliers_100} ({num_outliers_100 / len(df) * 100:.4f} %)")
    
    print(f"--------------------------------------------")

def generate_report(df: pd.DataFrame):
    """Generate the distribution of num_labels and report number of outliers"""
    
    fig = px.histogram(df.num_labels, x="num_labels", nbins=20)
    fig.show()
    
    num_outliers_50 = len(df[df.num_labels > 50])
    num_outliers_100 = len(df[df.num_labels > 100])
    print(f"Total number of samples: {len(df)}")
    print(f"# of samples with 50+ labels: {num_outliers_50} ({num_outliers_50 / len(df) * 100:.4f} %)")
    print(f"# of samples with 100+ labels: {num_outliers_100} ({num_outliers_100 / len(df) * 100:.4f} %)")
    print(f"--------------------------------------------")


if __name__ == "__main__":
    data_path = "/GET/data/msnbc.jsonl"
    visualize_data(data_path)