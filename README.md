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

## License
See the [LICENSE](LICENSE) file for details.