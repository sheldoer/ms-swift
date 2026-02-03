# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass
from typing import Optional

import json

from swift.utils import add_version_to_work_dir, get_logger, init_process_group, is_last_rank, to_abspath
from .megatron_base_args import MegatronBaseArguments

logger = get_logger()


@dataclass
class MegatronSftArguments(MegatronBaseArguments):
    add_version: bool = True
    load_args: bool = False
    # FieldTuneMixed: use same-dir mixed_dataset.FieldTuneMixedDataset (multi-dir blend + transforms).
    # When True, --dataset must be a directory (data_dir); data is loaded via FieldTuneMixedDataset.build_dataset().
    use_field_tune_mixed_dataset: bool = False
    # Path to weight_config JSON or JSON string: {"subdir_name": weight, ...} for interleave sampling.
    weight_config: Optional[str] = None
    # FieldTuneMixed optional: data_format, xiaoai_field, sharegpt, iot_seq (see mixed_dataset.FieldTuneMixedDataset).
    dataset_type: Optional[str] = "field_tune"
    concat_samples: Optional[bool] = True
    dialog_construct_method: Optional[str] = None
    dialog_loss_calc_part: Optional[str] = None
    dialog_sample_strategy: Optional[str] = None
    share_gpt_loss_calc_part: Optional[str] = None


    def _init_save(self):
        init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)
        if self.save is None:
            self.save = f'megatron_output/{self.model_suffix}'
        self.save = to_abspath(self.save)
        if self.add_version:
            self.save = add_version_to_work_dir(self.save)
            logger.info(f'args.save: {self.save}')
        if is_last_rank():
            os.makedirs(self.save, exist_ok=True)

    def _init_ckpt_dir(self, adapters=None):
        super()._init_ckpt_dir(adapters)
        if self.ckpt_dir and self.model is None:
            args_path = os.path.join(self.ckpt_dir, 'args.json')
            if not os.path.exists(args_path):
                return
            with open(args_path, 'r', encoding='utf-8') as f:
                old_args = json.load(f)
            self.model = old_args.get('model')

    def __post_init__(self):
        self.load = to_abspath(self.load, check_path_exist=True)
        super().__post_init__()
        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '
                             'Please input the training dataset.')
        self._init_save()
        if self.tensorboard_dir is None and self.save is not None:
            self.tensorboard_dir = f'{self.save}/runs'
        self.tensorboard_dir = to_abspath(self.tensorboard_dir)
        if self.load is None and self.model is None and self.no_initialization:
            raise ValueError('You did not pass `--load/--model` to read weights, so you need to set '
                             '`--no_initialization false` to allow the model to initialize weights properly.')
