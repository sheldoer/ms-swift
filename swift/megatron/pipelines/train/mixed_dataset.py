#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Author wangsiwen@xiaomi.com
    @Date 2023/6/19 下午8:17
    @Describe
    @Version 1.0
"""
import os
import json
import copy
from collections import OrderedDict
import numpy as np

from typing import Optional, Union
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets
from transformers import set_seed

from . import data_transform

SEED = 42
set_seed(SEED)


DATA_TRANSFORM_MAP = {
    "xiaoai_field": data_transform.xiaoai_field_tune_transform,
    "xiaoai_vocab": data_transform.xiaoai_extend_vocab_transform,
    "xiaoai_zk": data_transform.xiaoai_field_tune_for_zk_transform,
    "zk_mr_wh": data_transform.zk_sft_wh_transform,
    "zk_mr_bj": data_transform.zk_sft_bj_transform,
    "zk_sft_offline": data_transform.zk_sft_offline_transform,
    "control_multi_round": data_transform.control_sft_transform,
    "control_sft": data_transform.control_sft_transform,
    "aicreative_sft": data_transform.aicreative_sft_transform,
    "cpt_query_sft": data_transform.cpt_query_sft_transform,
    "wudao": data_transform.wudao_transform,
    "firefly": data_transform.firefly_transform,
    "sharegpt": data_transform.share_gpt_transform,
    "wiki": data_transform.wudao_transform,
    "codeparrot": data_transform.codeparrot_transform,
    "skywork": data_transform.identity_transform,
    "general_sft": data_transform.general_sft_transform,
    "general_ability": data_transform.general_ability_transform,
    "general_chat": data_transform.general_chat_sft_transform,
    "default": data_transform.identity_transform
}

DATA_CONCAT_STRATEGY_MAP = {
    "wiki": "SPLIT",
    "wudao": "TRUNC_OR_PAD",
    "skywork": "TRUNC_OR_PAD",
    "codeparrot": "PAD",
    "xiaoai_field": "PAD",
    "firefly": "PAD",
    "sharegpt": "PAD",
    "iot_seq_cpt": "PAD"
}

DATA_TYPE_MAP = {
    "xiaoai_intent_corpus": "xiaoai_field",
    "xiaoai_dialogue_corpus": "xiaoai_field",
    "xiaoai_intent_dialogue_corpus": "xiaoai_field",
    "xiaoai_intent_analysis_corpus": "xiaoai_field",
    "xiaoai_dialogue_corpus_for_vocab": "xiaoai_vocab",
    "xiaoai_intent_corpus_for_zk": "xiaoai_zk",
    "zk_mr_wh": "zk_mr_wh",
    "zk_mr_bj": "zk_mr_bj",
    "zk_sft_offline": "zk_sft_offline",
    "control_multi_round": "control_multi_round",
    "control_sft": "control_sft",
    "aicreative_sft": "aicreative_sft",
    "wudao": "wudao",
    "firefly": "firefly",
    "zh_wiki": "wiki",
    "skywork": "skywork",
    "codeparrot-clean": "codeparrot",
    "general_sft": "general_sft",
    "general_ability": "general_ability",
    "sharegpt": "sharegpt",
    "iot_seq_align_ul": "iot_seq_align_ul",
    "general_chat": "general_chat",
}


BATCH_TRANSFORM_TYPE = []


# SUPPORTED_MODEL_TYPES = (
#     "llama", "llama2", "baichuan", "qwen", "milm", "milm_new", "chatglm2", "chatglm3", "skywork", "gemma"
# )

# NEED_ATTN_MASK_TYPES = (
#     "llama", "llama2", "baichuan", "qwen", "milm", "chatglm2", "chatglm3", "skywork", "gemma"
# )

NEED_POS_ID_TYPES = (
    "chatglm2", "chatglm3"
)


GIST_TOKEN = {
    "milm": "<|unused0|>"
}


RESERVED_COLUMNS = ("input_ids", "labels", "attention_mask", "position_ids")
NO_SAMPLE_COLUMNS = RESERVED_COLUMNS + ("label_flag",)


@dataclass
class WeightDatasetConfig:
    dataset_name: str
    data_dir: str
    transform_type: str
    sample_ratio: float = field(default=1.0)
    dataset: Optional[Union[Dataset, IterableDataset]] = None


class MixedDataset:
    """
    Get a dataset using huggingface dataset tools.

    """

    def __init__(self, logger, data_dir, tokenizer,
                 max_length=256, streaming=True, weight_config=None,
                 local_rank=-1, default_extract_column="text", is_train=True,
                 model_type="llama", stopping_strategy=None, concat_samples=False,
                 mix_at_eval=True, use_shuffle=True, do_not_tokenize=False, data_num_proc=16,
                 use_chat_template=False, data_split_info="train", change_keywords_type=None):
        self.logger = logger
        self.data_dir = data_dir
        self.weight_dataset_config = []
        self.get_weight_config_for_datasets(weight_config)
        if not self.weight_dataset_config:
            self.weight_dataset_config.append(
                WeightDatasetConfig(dataset_name="default", data_dir=self.data_dir, transform_type="default")
            )
        self.tokenizer = tokenizer
        # Causal LM 训练使用左填充，pad_label_ids 依赖此设置
        if getattr(self.tokenizer, 'padding_side', None) != 'left':
            self.tokenizer.padding_side = 'left'
        self.max_length = max_length
        self.streaming = streaming
        self.default_extract_column = default_extract_column
        self.local_rank = local_rank
        self.is_train = is_train
        self.model_type = model_type
        self.stopping_strategy = stopping_strategy
        self.concat_samples = concat_samples
        self.mix_at_eval = mix_at_eval
        self.use_shuffle = use_shuffle
        self.do_not_tokenize = do_not_tokenize
        self.data_num_proc = data_num_proc
        self.use_chat_template = use_chat_template
        self.data_split_info = "train"
        if not self.streaming and self.is_train:
            self.data_split_info = data_split_info
        self.change_keywords_type = change_keywords_type

    def get_weight_config_for_datasets(self, weight_config):
        weight_config_json = {}
        if isinstance(weight_config, dict):
            weight_config_json = weight_config
        elif isinstance(weight_config, str):
            if os.path.isfile(weight_config):
                with open(weight_config) as f:
                    weight_config_json = json.load(f)
            else:
                try:
                    weight_config_json = json.loads(weight_config)
                except Exception as e:
                    weight_config_json = {}

        for sub_dir_name in weight_config_json:
            if not os.path.isdir(os.path.join(self.data_dir, sub_dir_name)):
                self.logger.info(f"sub dir {sub_dir_name} does not exist, and the weight config will set to the defaults.")
                weight_config_json = {}
                break

        if not weight_config_json:
            sub_dir_list = os.listdir(self.data_dir)
            if all(os.path.isdir(os.path.join(self.data_dir, sub_dir)) for sub_dir in sub_dir_list):
                weight_config_json = {sub_dir: 1 for sub_dir in sub_dir_list}

        total_weight = sum(weight_config_json.values())
        for sub_dir, weight in weight_config_json.items():
            transform_type = "default"
            if sub_dir in DATA_TYPE_MAP:
                transform_type = DATA_TYPE_MAP[sub_dir]
            elif sub_dir in DATA_TRANSFORM_MAP:
                transform_type = sub_dir
            else:
                sub_dir_fields = sub_dir.split("_")
                for i in range(1, len(sub_dir_fields)):
                    crt_prefix = "_".join(sub_dir_fields[:-i])
                    if crt_prefix in DATA_TYPE_MAP:
                        transform_type = DATA_TYPE_MAP[crt_prefix]
                        break

            self.weight_dataset_config.append(
                WeightDatasetConfig(
                    dataset_name=sub_dir,
                    data_dir=os.path.join(self.data_dir, sub_dir),
                    transform_type=transform_type,
                    sample_ratio=weight / total_weight
                )
            )

    def tokenize(self, sample):
        pre_text = sample.get("pre_text", "")
        post_text = sample.get("post_text")

        input_ids = sample.get("input_ids", None)
        label_ids = sample.get("labels", None)

        mask_info = sample.get("mask", None)
        label_flag = sample.get("label_flag", None)

        # assert self.model_type in SUPPORTED_MODEL_TYPES, f"model type {self.model_type} has not been supported!"

        if input_ids is None or label_ids is None:
            input_ids, pre_text_len = self.get_input_ids(pre_text, post_text)
            label_ids = self.get_label_ids(input_ids, pre_text + post_text, pre_text_len, label_flag=label_flag)

        label_ids = self.pad_label_ids(label_ids)

        feature = {
            "input_ids": input_ids[:self.max_length],
            "labels": label_ids[:self.max_length]
        }

        attention_mask = self.get_attention_masks(input_ids, pre_text + post_text, mask_info=mask_info)
        feature["attention_mask"] = attention_mask[:self.max_length]
        if self.model_type in NEED_POS_ID_TYPES:
            position_ids = self.get_position_ids(input_ids)
            feature["position_ids"] = position_ids[:self.max_length]
        return feature

    def get_input_ids(self, pre_text, post_text):
        """
        Get input ids by tokenization.

        """
        pre_text_len = len(self.tokenizer.encode(pre_text))
        tokenize_info = self.tokenizer(pre_text + post_text)
        input_ids = tokenize_info["input_ids"]

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and input_ids[-1] != eos_token_id:
            input_ids = input_ids + [eos_token_id]

        return input_ids, pre_text_len

    def get_label_ids(self, input_ids, text, pre_text_len, label_flag=None):
        """
        Get label ids.

        """
        label_ids = copy.deepcopy(input_ids)

        if not label_flag:
            label_ids = [-100] * pre_text_len + label_ids[pre_text_len:]
        else:
            label_range_list = self.get_mask_range_list(label_flag)
            new_label_flag, pre_accu_len = self.get_token_mask_from_char_mask(text, label_range_list)
            if len(new_label_flag) <= len(input_ids):
                new_label_flag.extend([1] * (len(input_ids) - pre_accu_len))
            else:
                new_label_flag = [1] * len(input_ids)
                self.logger.warning(f"desert this sample because the label flag length is too long: "
                                    f"the text is {text}, raw label flag is {label_flag}")

            label_ids = [-100 if new_label_flag[i] == 1 else label_ids[i] for i in range(len(new_label_flag))]

        return label_ids

    def pad_label_ids(self, label_ids, padding_token_id=-100):
        """
        Padding label ids.

        """
        if len(label_ids) < self.max_length:
            padding_side = self.tokenizer.padding_side
            assert padding_side == "left", "only support left padding!"
            remainder = [padding_token_id] * (self.max_length - len(label_ids))
            label_ids = remainder + label_ids
        return label_ids

    def get_attention_masks(self, input_ids, text, mask_info=None):
        """
        Get attention masks.

        """
        attention_mask = [1] * len(input_ids)
        if mask_info:
            mask_range_list = self.get_mask_range_list(mask_info)
            attention_mask, pre_accu_len = self.get_token_mask_from_char_mask(text, mask_range_list)
            attention_mask += [1] * (len(input_ids) - pre_accu_len)
        return attention_mask

    def get_position_ids(self, input_ids):
        """
        Get position ids.

        """
        position_ids = list(range(len(input_ids)))
        return position_ids

    def get_token_mask_from_char_mask(self, text, mask_range_list):
        mask = []
        pre_accu_len = 0
        fake_text = self.tokenizer.pad_token
        fake_len = len(self.tokenizer.encode(fake_text))
        crt_start = 0
        for mask_start, mask_end in mask_range_list:
            if crt_start == 0:
                crt_token_len = len(self.tokenizer.encode(text[crt_start: mask_start]))
            else:
                crt_token_len = len(self.tokenizer.encode(fake_text + text[crt_start: mask_start])) - fake_len
            mask.extend([1] * crt_token_len)
            pre_accu_len += crt_token_len

            if mask_start == 0:
                crt_token_len = len(self.tokenizer.encode(text[mask_start: mask_end]))
            else:
                crt_token_len = len(self.tokenizer.encode(fake_text + text[mask_start: mask_end])) - fake_len
            mask.extend([0] * crt_token_len)
            pre_accu_len += crt_token_len
            crt_start = mask_end
        return mask, pre_accu_len

    def get_mask_range_list(self, mask_info):
        mask_range_list = []
        left, right = None, None
        if mask_info:
            for idx, mask_flag in enumerate(mask_info):
                if mask_flag == 1 and left is None:
                    left = idx
                if mask_flag == 0 and left is not None:
                    right = idx
                    mask_range_list.append((left, right))
                    left = None
        if left is not None:
            mask_range_list.append((left, len(mask_info)))
        return mask_range_list

    def get_extra_args_for_trans(self, dataset_info):
        return {}

    def get_stopping_strategy(self):
        if self.stopping_strategy not in ("all_exhausted", "first_exhausted"):
            self.stopping_strategy = "all_exhausted"

    def get_columns_to_remove(self, dataset, reserved_columns=None):
        if self.streaming:
            first_row = list(dataset.take(1))[0]
            columns_to_remove = list(dataset.take(1))[0].keys()
        else:
            first_row = dataset[0]
            columns_to_remove = dataset.column_names
        columns_to_remove = list(columns_to_remove)

        if reserved_columns:
            reserved_columns = list(reserved_columns)
            if "label_flag" in reserved_columns and "label_flag" in columns_to_remove:
                label_flag_field = first_row.get("label_flag")
                if label_flag_field is None or (
                        isinstance(label_flag_field, list) and (not label_flag_field or label_flag_field[0] is None)):
                    reserved_columns.remove("label_flag")

            columns_to_remove = [col for col in columns_to_remove if col not in reserved_columns]
        return columns_to_remove

    def get_sample_records(self, dataset, sample_cnt=3):
        """
        get sample records for debug.

        """
        records = []
        if self.streaming:
            for record in dataset.take(sample_cnt):
                records.append(OrderedDict((k, v) for k, v in record.items() if k not in NO_SAMPLE_COLUMNS))
        else:
            records = [OrderedDict() for i in range(sample_cnt)]
            for k, v_list in dataset[:sample_cnt].items():
                if k in NO_SAMPLE_COLUMNS:
                    continue
                for v, record in zip(v_list, records):
                    record[k] = v
        return records

    def build_dataset(self):
        to_del_columns_after_tokenize = []
        optional_num_proc_kwargs = {}
        if not self.streaming:
            optional_num_proc_kwargs["num_proc"] = self.data_num_proc
        for dataset_info in self.weight_dataset_config:
            dataset_info.dataset = load_dataset(
                "json", data_dir=dataset_info.data_dir, split=self.data_split_info, streaming=self.streaming)
            if self.use_shuffle:
                if self.streaming:
                    dataset_info.dataset = dataset_info.dataset.skip(1)  # avoid shuffling files
                    dataset_info.dataset = dataset_info.dataset.shuffle(buffer_size=10_000, seed=SEED)
                else:
                    dataset_info.dataset = dataset_info.dataset.shuffle(seed=SEED)

            trans_kwargs = {}
            if dataset_info.transform_type == "default":
                trans_kwargs["column"] = self.default_extract_column
            trans_kwargs.update(self.get_extra_args_for_trans(dataset_info))
            if self.use_chat_template:
                trans_kwargs["use_chat_template"] = True
            trans_func = DATA_TRANSFORM_MAP.get(dataset_info.transform_type)
            batched = False
            if dataset_info.transform_type in BATCH_TRANSFORM_TYPE:
                batched = True
            dataset_info.dataset = dataset_info.dataset.map(
                trans_func, fn_kwargs=trans_kwargs, batched=batched,
                remove_columns=self.get_columns_to_remove(dataset_info.dataset),
                **optional_num_proc_kwargs)

            if self.is_train and self.concat_samples:
                concat_strategy = DATA_CONCAT_STRATEGY_MAP.get(dataset_info.transform_type, "TRUNC")
                dataset_info.dataset = dataset_info.dataset.map(
                    data_transform.concat_transform,
                    fn_kwargs={"tokenizer": self.tokenizer,
                               "max_seq_len": self.max_length,
                               "concat_strategy": concat_strategy},
                    batched=True,
                    remove_columns=self.get_columns_to_remove(dataset_info.dataset, reserved_columns=("post_text", "label_flag")),
                    **optional_num_proc_kwargs
                )

            if not to_del_columns_after_tokenize:
                to_del_columns_after_tokenize = self.get_columns_to_remove(
                    dataset_info.dataset, reserved_columns=RESERVED_COLUMNS)

            if self.local_rank == 0:
                records = self.get_sample_records(dataset_info.dataset, sample_cnt=3)
                self.logger.info(f"samples for dataset {dataset_info.data_dir}:\n"
                                 f"{json.dumps(records, ensure_ascii=False, indent=4)}")

        if self.is_train or self.mix_at_eval:
            self.get_stopping_strategy()
            interleaved_dataset = interleave_datasets(
                [dataset_info.dataset for dataset_info in self.weight_dataset_config],
                probabilities=[dataset_info.sample_ratio for dataset_info in self.weight_dataset_config],
                stopping_strategy=self.stopping_strategy,
            )

            # for debug
            if self.local_rank == 0:
                self.logger.info(f"==============sample ratio: "
                                 f"{[(d.data_dir, d.sample_ratio) for d in self.weight_dataset_config]}")
                records = self.get_sample_records(interleaved_dataset, sample_cnt=20)
                self.logger.info(f"samples for interleaved dataset:\n"
                                 f"{json.dumps(records, ensure_ascii=False, indent=4)}")

            if not self.do_not_tokenize:
                interleaved_dataset = interleaved_dataset.map(
                    self.tokenize, remove_columns=to_del_columns_after_tokenize, **optional_num_proc_kwargs
                ).filter(
                    lambda f: not all(e == -100 for e in f["labels"])
                )
        else:
            if not self.do_not_tokenize:
                for dataset_info in self.weight_dataset_config:
                    dataset_info.dataset = dataset_info.dataset.map(
                        self.tokenize, remove_columns=to_del_columns_after_tokenize, **optional_num_proc_kwargs
                    ).filter(
                        lambda f: not all(e == -100 for e in f["labels"])
                    )
            interleaved_dataset = OrderedDict((dataset_info.dataset_name, dataset_info.dataset)
                                              for dataset_info in self.weight_dataset_config)

        return interleaved_dataset


class FieldTuneMixedDataset(MixedDataset):
    """
    Get a dataset for xiaoai field tune using huggingface dataset tools.
    Aligned with ai_nlu FieldTuneMixedDataset: xiaoai_field, sharegpt
    """
    def __init__(self, logger, data_dir, tokenizer, max_length=256, streaming=True, weight_config=None, local_rank=-1,
                 default_extract_column="text", is_train=True, model_type="llama",
                 stopping_strategy=None, concat_samples=False, mix_at_eval=True,
                 use_shuffle=True, do_not_tokenize=False, data_num_proc=16, use_chat_template=False,
                 data_format="lm", xiaoai_multi_task=False, dialog_construct_method="two_role",
                 dialog_loss_calc_part="all", dialog_sample_strategy="no",
                 share_gpt_loss_calc_part="all",
                 **kwargs):
        super().__init__(logger, data_dir, tokenizer, max_length, streaming, weight_config, local_rank,
                         default_extract_column, is_train, model_type, stopping_strategy, concat_samples,
                         mix_at_eval, use_shuffle, do_not_tokenize, data_num_proc, use_chat_template)
        self.data_format = data_format

        # xiaoai_field
        self.xiaoai_multi_task = xiaoai_multi_task
        self.dialog_construct_method = dialog_construct_method
        self.dialog_loss_calc_part = dialog_loss_calc_part
        self.dialog_sample_strategy = dialog_sample_strategy

        # share_gpt
        self.share_gpt_loss_calc_part = share_gpt_loss_calc_part


    def get_stopping_strategy(self):
        if not self.stopping_strategy:
            self.stopping_strategy = "all_exhausted"
            if not self.is_train:
                self.stopping_strategy = "first_exhausted"

    def get_extra_args_for_trans(self, dataset_info):
        trans_kwargs = {"data_format": self.data_format}
        if dataset_info.transform_type == "xiaoai_field":
            trans_kwargs["xiaoai_multi_task"] = self.xiaoai_multi_task
            trans_kwargs["dialog_construct_method"] = self.dialog_construct_method
            trans_kwargs["dialog_loss_calc_part"] = self.dialog_loss_calc_part
            trans_kwargs["dialog_sample_strategy"] = self.dialog_sample_strategy
        elif dataset_info.transform_type == "sharegpt":
            trans_kwargs["share_gpt_loss_calc_part"] = self.share_gpt_loss_calc_part
        return trans_kwargs


class ZKFineTuneMixedDataset(MixedDataset):
    """
    Get a dataset for zhongkong fine tune using huggingface dataset tools.

    """
    def __init__(self, logger, data_dir, tokenizer, max_length=256, streaming=True, weight_config=None, local_rank=-1,
                 default_extract_column="text", is_train=True, model_type="llama", stopping_strategy=None,
                 concat_samples=False, mix_at_eval=True, use_shuffle=True, do_not_tokenize=False, data_num_proc=16,
                 use_chat_template=False, output_context_flag=False, repeat_mask_ratio=0, nonrepeat_mask_ratio=0,
                 perturb_domain_ratio=0, history_loss_ratio=0, only_use_last_round=False, use_gist=False, **kwargs):
        super().__init__(logger, data_dir, tokenizer, max_length, streaming, weight_config, local_rank,
                         default_extract_column, is_train, model_type, stopping_strategy, concat_samples, mix_at_eval,
                         use_shuffle, do_not_tokenize, data_num_proc, use_chat_template)
        self.stopping_strategy = "all_exhausted"
        self.output_context_flag = output_context_flag
        self.repeat_mask_ratio = repeat_mask_ratio
        self.nonrepeat_mask_ratio = nonrepeat_mask_ratio
        self.perturb_domain_ratio = perturb_domain_ratio
        self.history_loss_ratio = history_loss_ratio
        self.only_use_last_round = only_use_last_round
        self.use_gist = use_gist

    def get_extra_args_for_trans(self, dataset_info):
        trans_kwargs = {}
        if self.output_context_flag:
            trans_kwargs["context"] = True
        trans_kwargs["repeat_mask_ratio"] = self.repeat_mask_ratio
        trans_kwargs["nonrepeat_mask_ratio"] = self.nonrepeat_mask_ratio
        trans_kwargs["perturb_domain_ratio"] = self.perturb_domain_ratio
        trans_kwargs["history_loss_ratio"] = self.history_loss_ratio
        trans_kwargs["is_training"] = self.is_train
        trans_kwargs["only_use_last_round"] = self.only_use_last_round
        if self.use_gist:
            assert self.model_type in GIST_TOKEN, f"model type {self.model_type} does not support gist token."
            trans_kwargs["gist_token"] = GIST_TOKEN[self.model_type]
        return trans_kwargs


class ControlCodeGenMixedDataset(MixedDataset):
    """
    Get a dataset for multi-round fine-tuning for the control domain.

    """
    def __init__(self, logger, data_dir, tokenizer, max_length=1024, streaming=True, weight_config=None, local_rank=-1,
                 default_extract_column="text", is_train=True, model_type="llama",
                 stopping_strategy=None, concat_samples=False, mix_at_eval=True, use_shuffle=True,
                 do_not_tokenize=False, data_num_proc=16, use_chat_template=False, data_split_info="train",
                 change_keywords_type=None,
                 use_gist=False, use_code_context=False, prompt_template=None,
                 keep_raw_sample=False, think_start_token="<think>", think_end_token="</think>",
                 use_think=True, add_think_tag_in_prompt=True, think_prefix_token="",
                 add_think_decision=False, rewrite_sample_use_think=0,
                 **kwargs):
        super().__init__(logger, data_dir, tokenizer, max_length, streaming, weight_config, local_rank,
                         default_extract_column, is_train, model_type, stopping_strategy, concat_samples,
                         mix_at_eval, use_shuffle, do_not_tokenize, data_num_proc, use_chat_template,
                         data_split_info, change_keywords_type)
        self.stopping_strategy = "all_exhausted"
        self.use_gist = use_gist
        self.use_code_context = use_code_context
        self.prompt_template = prompt_template
        self.keep_raw_sample = keep_raw_sample
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.add_think_tag_in_prompt = add_think_tag_in_prompt
        self.use_think = use_think
        self.think_prefix_token = think_prefix_token
        self.add_think_decision = add_think_decision
        self.rewrite_sample_use_think = rewrite_sample_use_think

    def get_extra_args_for_trans(self, dataset_info):
        trans_kwargs = {}
        trans_kwargs["change_keywords_type"] = self.change_keywords_type
        trans_kwargs["use_code_context"] = self.use_code_context
        trans_kwargs["keep_raw_sample"] = self.keep_raw_sample
        trans_kwargs["tokenizer"] = self.tokenizer
        if self.use_gist:
            assert self.model_type in GIST_TOKEN, f"model type {self.model_type} does not support gist token."
            trans_kwargs["gist_token"] = GIST_TOKEN[self.model_type]
        if self.prompt_template:
            trans_kwargs["prompt_template"] = self.prompt_template
        if dataset_info.transform_type in ("general_sft", "general_chat"):
            trans_kwargs["think_start_token"] = self.think_start_token
            trans_kwargs["think_end_token"] = self.think_end_token
            trans_kwargs["add_think_tag_in_prompt"] = self.add_think_tag_in_prompt
            trans_kwargs["use_think"] = self.use_think
            trans_kwargs["think_prefix_token"] = self.think_prefix_token
            trans_kwargs["add_think_decision"] = self.add_think_decision
            trans_kwargs["rewrite_sample_use_think"] = self.rewrite_sample_use_think
        return trans_kwargs



def tokenizer_post_processor(tokenizer):
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]


class DummyDataset:
    def __init__(self, seq_length, sample_num, vocab_size, model_type="LLaMA"):
        self.seq_length = seq_length
        self.sample_num = sample_num
        self.vocab_size = vocab_size
        self.model_type = model_type

    def gen(self):
        for _ in range(self.sample_num):
            if self.model_type == "chatGLM":
                yield {
                    "input_ids":
                        np.concatenate(
                            [
                                [130001, 130004],
                                np.random.randint(low=0, high=self.vocab_size, size=self.seq_length-2, dtype=np.int64),
                            ]
                        ),
                    "label_ids": np.concatenate(
                            [
                                [130001, 130004],
                                np.random.randint(low=0, high=self.vocab_size, size=self.seq_length-2, dtype=np.int64),
                            ]
                        ),
                }
            else:
                yield {
                    "input_ids": np.random.randint(low=0, high=self.vocab_size, size=self.seq_length, dtype=np.int64),
                    "label_ids": np.random.randint(low=0, high=self.vocab_size, size=self.seq_length, dtype=np.int64)
                }

    def build_dataset(self):
        dataset = Dataset.from_generator(self.gen)
        return dataset
