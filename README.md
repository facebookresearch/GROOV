# GROOV: GeneRative Out-Of_Vocabulary tagging

This is a minimal codebase to reproduce data and models for the paper "Open Vocabulary Extreme Classification Using Generative Models"

## Data

To reproduce the AmazonCat-OV dataset:
1) Download raw text features for AmazonCat-13K from http://manikvarma.org/downloads/XC/XMLRepository.html
2) Format the data using utils in data_processing/extract_data.py
3) Create the shuffled data by running data_processing/create_amazoncat_ov.py

## Models
The three main steps to produce GROOV tagger models:

Finetune T5 on a dataset:
```
python finetune_s2s.py --train_file_path=<YOUR_TRAIN_FILE> --test_file_path=<YOUR_DEV_FILE> --train_batch_size=32 --eval_batch_size=32 --output_dir=test_run_results/t5_small_10ep --model_name_or_path t5-small --use_multisoftmax --data_parallel --save_after_every_eval --eval_every_k_epoch 5 --num_epochs 10
```

Run inference on a model:
```
python load_and_eval_s2s.py --train_file_path=<YOUR_TRAIN_FILE> --test_file_path=<YOUR_TEST_FILE> --output_dir=<OUTPUT_DIR> --eval_batch_size 10 --decode_beams 15
```

Compute PSP@K metrics on the result of inference:
```
python eval.py --guess <OUTPUT_DIR>/test_preds_sum_prob.jsonl --gold <YOUR_TEST_FILE> --ks 1,5,10,15
python eval_psp.py --train <YOUR_TRAIN_FILE> --gold <YOUR_TEST_FILE> --guess <OUTPUT_DIR>/test_preds_sum_prob.jsonl
```

# GET
GET is an entity tagging model that extracts set of entities without mention supervision. 

## Requirements
```
python == 3.7
pytorch == 1.9.1
transformers == 4.9.1
```
## Usage
### Download model and data
The pretrained GET model can be downloaded [here](https://dl.fbaipublicfiles.com/groov/get_model.tar.gz). To replicate our experiments in GET paper, download the [training data and WNED benchmark](https://dl.fbaipublicfiles.com/groov/get_data.tar.gz). 

### Configuration
Set the path to model checkpoint and dataset in `local_configs.py`

### Train the model
Finetune GET model on Wikipedia abstracts and AIDA data:
```bash
LOCAL_DATA_DIR = "../GET/data"

python finetune_s2s.py --train_file_path pretrain_data/small_train.jsonl
                       --test_file_path pretrain_data/wiki_abstract_aida_dev.jsonl
                       --output_dir <experiment_name>
                       --model_name_or_path t5-base
                       --train_batch_size 16
                       --eval_batch_size 2
                       --num_epochs 50
                       --max_i_length 512
                       --max_o_length 512
                       --data_parallel
```
### Evaluation
Generate the prediction for AIDA test data using constrained beam search:

```bash
LOCAL_DATA_DIR = "../GET/data"
OUTPUT_DIR = "../GET"

python load_and_eval_s2s.py --output_dir experiments/<experiment_name> 
                            --model_name_or_path t5-base 
                            --test_file_path AIDA/aida_test_dataset.jsonl
                            --decode_on_lattice 
                            --decode_beams 5 
                            --label_set_file entities.json
                            --dataset_name aida_test
```

Compute the evaluation metrics on AIDA:
```bash
python eval.py --guess <OUTPUT_DIR>/test_preds_naive_lattice_aida_test.jsonl
               --gold ../GET/data/AIDA/aida_test_dataset.jsonl
```

## License
See the [LICENSE](LICENSE) file for details.
