# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import asdict
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.utils import is_torch_npu_available

from swift.megatron.arguments import MegatronSftArguments
from swift.megatron.trainers import MegatronEmbeddingTrainer, MegatronRerankerTrainer, MegatronTrainer
from swift.megatron.utils import get_padding_to
from swift.pipelines import SwiftSft
from swift.utils import get_logger, is_last_rank, plot_images
from .utils import build_streaming_dataloader

if is_torch_npu_available():
    # Enable Megatron on Ascend NPU
    from mindspeed.megatron_adaptor import repatch
else:
    repatch = None

logger = get_logger()


# Wrapper so that FieldTuneMixed-built dataset yields dicts compatible with template.data_collator
# (input_ids, labels, position_ids). No LazyLLMDataset / template.encode needed.
class _PreTokenizedSwiftDataset(torch.utils.data.Dataset):
    """Wraps a HF/Interleave dataset from FieldTuneMixedDataset.build_dataset() so each
    __getitem__ returns {input_ids, labels, position_ids} for Swift Megatron data_collator.
    """

    def __init__(self, hf_dataset: Any) -> None:
        self._ds = hf_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._ds[idx]
        input_ids = row['input_ids']
        labels = row['labels']
        if hasattr(input_ids, 'tolist'):
            input_ids = input_ids.tolist()
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        n = len(input_ids)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': list(range(n)),
        }


def _build_field_tune_mixed_datasets(args: MegatronSftArguments, template) -> tuple:
    """Build train (and optionally val) dataset using same-dir FieldTuneMixedDataset
    (swift/megatron/pipelines/train/mixed_dataset.py).
    Returns (train_dataset, val_dataset) where train_dataset is _PreTokenizedSwiftDataset
    compatible with template.data_collator. val_dataset is None or same type.
    """
    from .mixed_dataset import FieldTuneMixedDataset
    data_dir = args.dataset[0] if isinstance(args.dataset, list) else args.dataset
    if not os.path.isdir(data_dir):
        raise ValueError(
            f'use_field_tune_mixed_dataset requires --dataset to be a directory (data_dir). '
            f'Got: {data_dir}'
        )
    tokenizer = template.processor
    if tokenizer is None:
        raise ValueError('FieldTuneMixed requires template.processor (HF tokenizer).')
    max_length = getattr(args, 'max_length', None) or getattr(args, 'seq_length', 2048)
    # Align with FieldTuneMixedDataset in same dir (mixed_dataset.py); pass optional kwargs from args.
    mixed = FieldTuneMixedDataset(
        logger=logger,
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        streaming=False,
        weight_config=getattr(args, 'weight_config', None),
        local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        default_extract_column='text',
        is_train=True,
        model_type='qwen3',
        stopping_strategy='all_exhausted',
        concat_samples=False,
        mix_at_eval=True,
        use_shuffle=args.dataset_shuffle,
        do_not_tokenize=False,
        data_num_proc=getattr(args, 'dataset_num_proc', 4),
        use_chat_template=getattr(template, 'use_chat_template', False),
        data_format=getattr(args, 'data_format', None) or 'lm',
        xiaoai_multi_task=getattr(args, 'xiaoai_multi_task', False),
        dialog_construct_method=getattr(args, 'dialog_construct_method', None) or 'three_role_with_tts',
        dialog_loss_calc_part=getattr(args, 'dialog_loss_calc_part', None) or 'all',
        dialog_sample_strategy=getattr(args, 'dialog_sample_strategy', None) or 'no',
        share_gpt_loss_calc_part=getattr(args, 'share_gpt_loss_calc_part', None) or 'assistant',
    )
    interleaved = mixed.build_dataset()
    # interleaved may be InterleaveDataset or HfDataset; ensure we have __len__ and __getitem__
    base_ds = _PreTokenizedSwiftDataset(interleaved)
    val_dataset = None
    if args.split_dataset_ratio and args.split_dataset_ratio > 0 and len(base_ds) > 0:
        from torch.utils.data import Subset
        n = len(base_ds)
        val_size = max(1, int(n * args.split_dataset_ratio))
        train_size = n - val_size
        train_dataset = Subset(base_ds, range(train_size))
        val_dataset = Subset(base_ds, range(train_size, n))
        val_dataset.dataset_type = 'validation'
    else:
        train_dataset = base_ds
    return train_dataset, val_dataset



class MegatronSft(SwiftSft):
    args_class = MegatronSftArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        if args.task_type == 'embedding':
            return MegatronEmbeddingTrainer(self.args, self.template)
        elif args.task_type in {'reranker', 'generative_reranker'}:
            return MegatronRerankerTrainer(self.args, self.template)
        else:
            return MegatronTrainer(self.args, self.template)

    def __init__(self, args: Optional[Union[List[str], MegatronSftArguments]] = None) -> None:
        self.train_msg = {}
        super(SwiftSft, self).__init__(args)
        args = self.args
        if repatch is not None:
            if args.attention_backend != 'local':
                # MindSpeed requires passing `use_flash_attn` to Megatron
                # to enable flash attention on Ascend NPU.
                args.use_flash_attn = True
            megatron_args = asdict(self.args)
            repatch(megatron_args)
        template_cls = args.template_meta.template_cls
        if args.model_meta.is_multimodal and template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(**kwargs, download_model=args.load is None)
        self._prepare_template()
        args.init_model_args(self.tokenizer, self.processor.model_info.config)
        args.save_args(args.save)
        self.template.use_megatron = True
        self.trainer = self.prepare_trainer()

    def _get_data_collator(self):
        data_collator = self.template.data_collator
        padding_to = get_padding_to(self.args)
        logger.info(f'padding_to: {padding_to}')
        data_collator = partial(data_collator, padding_to=padding_to)
        return data_collator

    def _prepare_dataset(self):
        args = self.args
        if getattr(args, 'use_field_tune_mixed_dataset', False):
            logger.info('Using FieldTuneMixed dataset (same-dir mixed_dataset.FieldTuneMixedDataset).')
            train_dataset, val_dataset = _build_field_tune_mixed_datasets(args, self.template)
            return train_dataset, val_dataset
        return super()._prepare_dataset()

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        data_collator = self._get_data_collator()

        if args.streaming:
            train_dataset = build_streaming_dataloader(args, train_dataset, data_collator)
            if val_dataset is not None:
                val_dataset = build_streaming_dataloader(args, val_dataset, data_collator)

        try:
            self.trainer.train(train_dataset, val_dataset, data_collator)
        finally:
            # Visualization
            if is_last_rank():
                images_dir = os.path.join(args.save, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, args.tensorboard_dir)


def megatron_sft_main(args: Optional[Union[List[str], MegatronSftArguments]] = None):
    return MegatronSft(args).main()
