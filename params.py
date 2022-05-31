# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os


class ArgumentsS2S(argparse.ArgumentParser):
    def __init__(
        self,
        add_s2s_args=True,
        decode_mode=False,
        description="S2S parser",
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler="resolve",
            formatter_class=argparse.HelpFormatter,
            add_help=add_s2s_args,
        )

        if add_s2s_args:
            self.add_s2s_args()

        if decode_mode:
            self.add_decode_args()

    def add_s2s_args(self):
        parser = self.add_argument_group("Common Arguments")

        # Directories
        parser.add_argument(
            "--model_name_or_path",
            default="t5-large",
            type=str,
            help="Pretrained model name or path",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="tmp",
            help="Model output path",
        )
        parser.add_argument(
            "--train_file_path",
            type=str,
            help="Path of the training file relative to the working folder defined in local_configs.py"
        )
        parser.add_argument(
            "--test_file_path",
            type=str,
            help="Path of the test file relative to the working folder defined in local_configs.py"
        )

        # GPU use
        parser.add_argument(
            "--device",
            default="cuda",
            type=str,
            help="Device: CPU or CUDA",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Use torch.DataParallel(). Don't set device when using this!"
        )

        # Model settings
        parser.add_argument(
            "--max_i_length",
            default=256,
            type=int,
            help="Max input length",
        )
        parser.add_argument(
            "--max_o_length",
            default=128,
            type=int,
            help="Max output length",
        )
        parser.add_argument(
            "--single_label",
            action="store_true",
        )
        parser.add_argument(
            "--use_multisoftmax",
            action="store_true",
        )

        # Train / eval settings
        parser.add_argument(
            "--train_batch_size",
            default=2,
            type=int,
        )
        parser.add_argument(
            "--backward_freq",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--learning_rate",
            default=2e-4,
            type=float,
        )
        parser.add_argument(
            "--num_epochs",
            default=200,
            type=int,
        )
        parser.add_argument(
            "--eval_batch_size",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--eval_every_k_epoch",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--eval_sampling_rate",
            default=None,
            type=float,
            help="Only use this portion of the eval set.", 
        )
        parser.add_argument(
            "--print_freq",
            default=20,
            type=int,
        )
        parser.add_argument(
            "--save_after_every_eval",
            action="store_true",
        )

        # Misc
        parser.add_argument(
            "--replace_underscores",
            default=True,
            type=bool,
        )
        parser.add_argument(
            "--use_proxy",
            action="store_true",
        )

    def add_decode_args(self):
        parser = self.add_argument_group("Decoding-related Arguments")
        parser.add_argument(
            "--decode_on_lattice",
            action="store_true",
        )
        parser.add_argument(
            "--decode_beams",
            default=32,
            type=int,
        )
        parser.add_argument(
            "--pred_file_prefix",
            default="test_preds",
            type=str,
        )
