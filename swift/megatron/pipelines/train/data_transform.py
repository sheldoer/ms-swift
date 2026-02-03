#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Author wangsiwen@xiaomi.com
    @Date 2023/6/19 下午8:56
    @Describe 
    @Version 1.0
"""
import copy
import json
import random
import math
import re
import logging
from pyexpat.errors import messages

from jinja2 import Template, Environment, meta

from .data_transform_config import (
    XIAOAI_GLM_ROUND_ONE_PROMPTS, XIAOAI_FIELD_TUNE_PRE_PROMPTS)
from .data_transform_config import ZK_FINE_TUNE_PROMPTS, ZK_BJ_HIST_REGX
from .data_transform_config import DEFAULT_CONTROL_SFT_PROMPTS
from .data_transform_config import (INTENT_ANALYSIS_CONFIG, REWRITE_INTENT_ANALYSIS_CONFIG)
from .data_transform_config import COMPLEX_EPISODE_JUDGE_PROMPT_TEMPLATE
from .data_transform_helper import (chat_template_wrapper, add_none_label_flag, retrieve,
                                    get_general_ability_prefix, keep_original_sample,
                                    change_sample_keywords)


random.seed(42)


def identity_transform(sample, **kwargs):
    if isinstance(sample, str):
        return {"pre_text": "", "post_text": sample}

    if "column" in kwargs:
        assert kwargs["column"] in sample, \
            f"specified column {kwargs['column']} is not in the sample keys, please check!"
        return {"pre_text": "", "post_text": sample[kwargs["column"]]}
    assert "text" in sample, \
        "either `column` arg is not specified or default column `text` is not found in the sample."
    return {"pre_text": "", "post_text": sample["text"]}


def xiaoai_extend_vocab_transform(sample, **kwargs):
    pre_text = ""
    post_text = ""
    dialogues = sample.get("dialogues", [])
    for single_round in dialogues:
        role = single_round["role"]
        msg = single_round["message"]
        if role not in ("小爱", "用户"):
            continue
        post_text += f"{role}：{msg}\n"
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


@add_none_label_flag
def xiaoai_field_tune_transform(sample, **kwargs):
    data_format = kwargs["data_format"]
    dialog_construct_method = kwargs.get("dialog_construct_method", "two_role")
    dialog_loss_calc_parts = kwargs.get("dialog_loss_calc_part", "all").split(",")
    dialog_sample_strategy = kwargs.get("dialog_sample_strategy", "no")

    role_map = {"query": "用户", "tts": "小爱", "analysis": "分析", "context": "背景"}

    if "concat_samples" in sample:
        samples = sample["concat_samples"]
    else:
        samples = [sample]

    dataset_type = samples[0].get("dataset", "dialogue")
    dialogues = []
    domain_list = []
    freq_list = []
    for sample in samples:
        dialogues += sample.get("dialogues", [])
        domains = sample.get("domains", "")
        domain_list += domains.split(",")
        freq_field = sample.get("freq", [])
        if isinstance(freq_field, list):
            freq_list = freq_field

    new_dialogues = []
    if "role" in dialogues[0]:    # old schema
        for i in range(len(dialogues) // 2):
            crt_dialog = {
                "query": dialogues[2 * i]["message"],
            }
            if dataset_type.startswith("intent_analysis"):
                crt_dialog["analysis"] = dialogues[2 * i + 1]["message"]
            else:
                crt_dialog["tts"] = dialogues[2 * i + 1]["message"]
            new_dialogues.append(crt_dialog)
    else:   # new schema
        new_dialogues = dialogues

    if len(new_dialogues) != len(freq_list):
        dialog_sample_strategy = "no"

    context_info = ""
    pre_text = XIAOAI_FIELD_TUNE_PRE_PROMPTS.get(dataset_type, "")
    post_text = ""
    if context_info:
        pre_text += f"背景信息：{context_info}\n"

    valid_role_list = ["query"]
    if dialog_construct_method == "two_role":
        if dataset_type.startswith("intent_analysis"):
            valid_role_list.append("analysis")
        else:
            valid_role_list.append("tts")
    elif dialog_construct_method == "three_role_with_tts":
        valid_role_list.extend(["analysis", "tts"])

    if data_format in ("last", "whole"):   # for evaluation only
        for dialog in new_dialogues[:-1]:
            for role in valid_role_list:
                pre_text += f"{role_map[role]}：\n{dialog[role]}\n"
        if data_format == "last":
            role = valid_role_list[0]
            pre_text += f"{role_map[role]}：\n{new_dialogues[-1][role]}\n"
            role = valid_role_list[1]
            pre_text += f"{role_map[role]}：\n"
            post_text += f"{new_dialogues[-1][role]}\n"
        else:
            for role in valid_role_list:
                pre_text += f"{role_map[role]}：\n{new_dialogues[-1][role]}\n"
        return {
            "pre_text": pre_text,
            "post_text": post_text
        }

    post_text = pre_text
    label_flag = [0] * len(pre_text)
    label_flag_by_role_strat = []
    label_flag_by_sample_strat = []
    for idx, dialog in enumerate(new_dialogues):
        keep_dialog = True
        if "dialog_mask" in sample and idx < len(sample["dialog_mask"]):
            cur_dialog_mask = sample["dialog_mask"][idx]
            if cur_dialog_mask == 1:
                keep_dialog = False

        if dialog_sample_strategy == "freq" and len(new_dialogues) > 1:
            crt_freq = freq_list[idx]
            drop_ratio = math.log(max(crt_freq - 4, 1)) * 0.1
            drop_ratio = min(0.95, drop_ratio)
            if random.random() < drop_ratio:
                keep_dialog = False

        for role in valid_role_list:
            crt_text_role_part = f"{role_map[role]}：\n"
            post_text += crt_text_role_part
            if dialog_loss_calc_parts[0] == "all":
                label_flag_by_role_strat += [1] * len(crt_text_role_part)
            else:
                label_flag_by_role_strat += [0] * len(crt_text_role_part)
            crt_text_msg_part = f"{dialog[role]}\n"
            post_text += crt_text_msg_part
            if dialog_loss_calc_parts[0] == "all" or role in dialog_loss_calc_parts:
                if role != "tts" or "tts_mask" not in sample:
                    label_flag_by_role_strat += [1] * len(crt_text_msg_part)
                elif sample["tts_mask"][idx] == 1:
                    label_flag_by_role_strat += [0] * len(crt_text_msg_part)
                else:
                    label_flag_by_role_strat += [1] * len(crt_text_msg_part)
            else:
                label_flag_by_role_strat += [0] * len(crt_text_msg_part)

            if keep_dialog:
                label_flag_by_sample_strat += [1] * len(crt_text_role_part + crt_text_msg_part)
            else:
                label_flag_by_sample_strat += [0] * len(crt_text_role_part + crt_text_msg_part)

    for label_1, label_2 in zip(label_flag_by_role_strat, label_flag_by_sample_strat):
        label_flag.append(label_1 & label_2)
    return {
        "pre_text": "",
        "post_text": post_text,
        "label_flag": label_flag
    }


def xiaoai_field_tune_for_zk_transform(sample, **kwargs):
    if "concat_samples" in sample:
        samples = sample["concat_samples"]
    else:
        samples = [sample]

    dialogues = []
    for sample in samples:
        dialogues += sample["messages"]
    pre_text = ZK_FINE_TUNE_PROMPTS["default"]
    post_text = ""
    for single_round in dialogues:
        post_text += f"{single_round['role']}：{single_round['content']}\n"
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


def intent_slots_single_round_eval_transform(sample, **kwargs):
    data_format = kwargs["data_format"]
    if data_format == "glm":
        pre_text = XIAOAI_GLM_ROUND_ONE_PROMPTS.get("intent_dialogue")
        if "{}" in pre_text:
            pre_text = pre_text.format("")
        pre_text += "[Round 2]\n\n问：{}\n\n答：".format(sample["query"])
    elif data_format == "lm":
        pre_text = XIAOAI_FIELD_TUNE_PRE_PROMPTS.get("intent_dialogue")
        if "{}" in pre_text:
            pre_text = pre_text.format("")
        pre_text += "用户：{}\n小爱：".format(sample["query"])
    else:
        pre_text = XIAOAI_FIELD_TUNE_PRE_PROMPTS.get("intent")
        if "{}" in pre_text:
            pre_text = pre_text.format("")
        pre_text += "用户：{}\n小爱：".format(sample["query"])
    return {
        "pre_text": pre_text
    }


def intent_slots_multi_round_eval_transform(sample, **kwargs):
    data_format = kwargs["data_format"]
    has_domain = kwargs.get("has_domain", False)
    history = json.loads(sample["history"])
    if len(history) > 3:
        history = history[-3:]
    if data_format == "glm":
        pre_text = XIAOAI_GLM_ROUND_ONE_PROMPTS.get("intent")
        if "{}" in pre_text:
            pre_text = pre_text.format("")
        idx = 2
        for diag_round in history:
            pre_text += "[Round {}]\n\n问：{}\n\n答：".format(idx, diag_round["query"])
            if has_domain:
                pre_text += "类别:{}，".format(diag_round["domain_name"])
            pre_text += "意图:{}".format(diag_round["intent_name"])
            if diag_round["slots"]:
                pre_text += "，槽位:{}".format(json.dumps(diag_round["slots"], ensure_ascii=False))
            pre_text += "\n\n"
            idx += 1
        pre_text += "[Round 2]\n\n问：{}\n\n答：".format(sample["query"])
    else:
        pre_text = XIAOAI_FIELD_TUNE_PRE_PROMPTS.get("intent")
        if "{}" in pre_text:
            pre_text = pre_text.format("")
        for diag_round in history:
            pre_text += "用户：{}\n小爱：".format(diag_round["query"])
            if has_domain:
                pre_text += "类别:{}，".format(diag_round["domain_name"])
            pre_text += "意图:{}".format(diag_round["intent_name"])
            if diag_round["slots"]:
                pre_text += "，槽位:{}".format(json.dumps(diag_round["slots"], ensure_ascii=False))
            pre_text += "\n"
        pre_text += "用户：{}\n小爱：".format(sample["query"])

    return {
        "pre_text": pre_text
    }


def zk_sft_wh_transform(sample, **kwargs):
    context_flag = kwargs.get("context_flag", False)
    is_training = kwargs.get("is_training", True)
    only_use_last_round = kwargs.get("only_use_last_round", False)
    repeat_mask_ratio = kwargs.get("repeat_mask_ratio", 0)
    nonrepeat_mask_ratio = kwargs.get("nonrepeat_mask_ratio", 0)
    perturb_domain_ratio = kwargs.get("perturb_domain_ratio", 0)
    history_loss_ratio = kwargs.get("history_loss_ratio", 0)

    gist_token = kwargs.get("gist_token", None)

    pre_text = ZK_FINE_TUNE_PROMPTS["default"]
    if context_flag:
        pre_text = ZK_FINE_TUNE_PROMPTS["context"]

    if gist_token:
        pre_text = gist_token
    mask = [0] * len(pre_text)
    label_flag = [0] * len(pre_text)
    dialogues = sample["messages"]
    domain_list = [dialogue["content"] for i, dialogue in enumerate(dialogues) if i % 2]
    mask_flag_list = []
    pre_domain = domain_list[-1]
    for domain in domain_list[-2::-1]:
        if domain == pre_domain:
            mask_flag = random.random() < repeat_mask_ratio
        else:
            mask_flag = random.random() < nonrepeat_mask_ratio
        pre_domain = domain
        mask_flag_list.append(mask_flag)
    mask_flag_list = mask_flag_list[::-1]

    if only_use_last_round:
        text = f"用户：{dialogues[-2]['content']}\n"
        mask += [0] * len(text)
        label_flag += [0] * len(text)
        pre_text += text
    else:
        round_cnt = 0
        for i, dialogue in enumerate(dialogues[:-1]):
            if i % 2 == 0:
                text = f"用户：{dialogue['content']}\n"
                mask_flag = False
                label_flag += [0] * len(text)
            else:
                perturb_flag = random.random() < perturb_domain_ratio
                original_domain = dialogue['content']
                changed_domain = original_domain
                if perturb_flag and is_training:
                    # candidate_domains = ZK_PERTURB_DOMAINS - {original_domain}
                    candidate_domains = ["问答", "闲聊", "其他"]
                    changed_domain = random.choice(list(candidate_domains))
                piece1, piece2, piece3 = "小爱：", changed_domain, "\n"
                text = piece1 + piece2 + piece3

                calc_loss_flag = random.random() < history_loss_ratio
                label_flag += [0] * len(piece1)
                if calc_loss_flag and not perturb_flag and is_training:
                    label_flag += [1] * len(piece2)
                else:
                    label_flag += [0] * len(piece2)
                label_flag += [0] * len(piece3)

                mask_flag = mask_flag_list[round_cnt] if is_training else False
                round_cnt += 1
            pre_text += text
            if mask_flag:
                mask += [1] * len(text)
            else:
                mask += [0] * len(text)
    text = "小爱："
    pre_text += text
    mask += [0] * len(text)
    label_flag += [0] * len(text)

    post_text = dialogues[-1]["content"] + "\n"
    mask += [0] * len(post_text)
    label_flag += [1] * len(post_text)
    feature = {
        "pre_text": pre_text,
        "post_text": post_text
    }
    if repeat_mask_ratio > 0 or nonrepeat_mask_ratio > 0:
        feature["mask"] = mask
    if history_loss_ratio > 0:
        feature["label_flag"] = label_flag
    return feature


def zk_sft_wh_arbitrator_transform(sample, **kwargs):
    context_flag = kwargs.get("context_flag", False)
    is_training = kwargs.get("is_training", True)
    only_use_last_round = kwargs.get("only_use_last_round", False)
    repeat_mask_ratio = kwargs.get("repeat_mask_ratio", 0)
    nonrepeat_mask_ratio = kwargs.get("nonrepeat_mask_ratio", 0)
    perturb_domain_ratio = kwargs.get("perturb_domain_ratio", 0)
    history_loss_ratio = kwargs.get("history_loss_ratio", 0)
    pre_text = ZK_FINE_TUNE_PROMPTS["default"]
    if context_flag:
        pre_text = ZK_FINE_TUNE_PROMPTS["context"]
    mask = [0] * len(pre_text)
    label_flag = [0] * len(pre_text)
    dialogues = sample["messages"]
    domain_list = [dialogue["content"] for i, dialogue in enumerate(dialogues) if i % 2]
    mask_flag_list = []
    pre_domain = domain_list[-1]
    for domain in domain_list[-2::-1]:
        if domain == pre_domain:
            mask_flag = random.random() < repeat_mask_ratio
        else:
            mask_flag = random.random() < nonrepeat_mask_ratio
        pre_domain = domain
        mask_flag_list.append(mask_flag)
    mask_flag_list = mask_flag_list[::-1]

    if only_use_last_round:
        text = f"用户:{dialogues[-2]['content']}\n"
        mask += [0] * len(text)
        label_flag += [0] * len(text)
        pre_text += text
    else:
        round_cnt = 0
        for i, dialogue in enumerate(dialogues[:-1]):
            if i % 2 == 0:
                text = f"用户:{dialogue['content']}\n"
                mask_flag = False
                label_flag += [0] * len(text)
            else:
                perturb_flag = random.random() < perturb_domain_ratio
                original_domain = dialogue['content']
                changed_domain = original_domain
                if perturb_flag and is_training:
                    # candidate_domains = ZK_PERTURB_DOMAINS - {original_domain}
                    candidate_domains = ["问答", "闲聊", "其他"]
                    changed_domain = random.choice(list(candidate_domains))
                piece1, piece2, piece3 = "小爱:", changed_domain, "\n"
                text = piece1 + piece2 + piece3

                calc_loss_flag = random.random() < history_loss_ratio
                label_flag += [0] * len(piece1)
                if calc_loss_flag and not perturb_flag and is_training:
                    label_flag += [1] * len(piece2)
                else:
                    label_flag += [0] * len(piece2)
                label_flag += [0] * len(piece3)

                mask_flag = mask_flag_list[round_cnt] if is_training else False
                round_cnt += 1
            pre_text += text
            if mask_flag:
                mask += [1] * len(text)
            else:
                mask += [0] * len(text)
    text = "小爱:"
    pre_text += text
    mask += [0] * len(text)
    label_flag += [0] * len(text)
    post_text = dialogues[-1]["content"]
    feature = {
        "pre_text": pre_text,
        "post_text": post_text
    }
    if repeat_mask_ratio > 0 or nonrepeat_mask_ratio > 0:
        feature["mask"] = mask
    if history_loss_ratio > 0:
        feature["label_flag"] = label_flag
    return feature


def zk_sft_bj_transform(sample, **kwargs):
    context_flag = kwargs.get("context_flag", False)
    is_training = kwargs.get("is_training", True)
    only_use_last_round = kwargs.get("only_use_last_round", False)
    repeat_mask_ratio = kwargs.get("repeat_mask_ratio", 0)
    nonrepeat_mask_ratio = kwargs.get("nonrepeat_mask_ratio", 0)
    perturb_domain_ratio = kwargs.get("perturb_domain_ratio", 0)
    pre_text = ZK_FINE_TUNE_PROMPTS["default"]
    if context_flag:
        pre_text = ZK_FINE_TUNE_PROMPTS["context"]
    mask = [0] * len(pre_text)
    history = []
    if sample["HISTORY"] != '':
        history = sample["HISTORY"].split(" \n ")
        if history[-1].endswith(" \n"):
            history[-1] = history[-1][:-2]

    extract_hist_list = []
    for hist in history:
        query = ""
        domain = ""
        for match in re.finditer(ZK_BJ_HIST_REGX, hist):
            query = match.groupdict()["query"]
            if query.endswith(" "):
                query = query[:-1]
            domain = match.groupdict()["domain"]
            if domain.endswith(" "):
                domain = domain[:-1]
        extract_hist_list.append((query, domain))

    crt_query = sample["QUERY"]
    crt_domain = sample["INTENTION"].split("\t")[-1]

    domain_list = [h[1] for h in extract_hist_list]
    mask_flag_list = []
    pre_domain = crt_domain
    for domain in domain_list[::-1]:
        if domain == pre_domain:
            mask_flag = random.random() < repeat_mask_ratio
        else:
            mask_flag = random.random() < nonrepeat_mask_ratio
        pre_domain = domain
        mask_flag_list.append(mask_flag)
    mask_flag_list = mask_flag_list[::-1]
    if not only_use_last_round:
        round_cnt = 0
        for query, domain in extract_hist_list:
            text = f"用户：{query}\n"
            pre_text += text
            mask += [0] * len(text)

            changed_domain = domain
            perturb_flag = random.random() < perturb_domain_ratio
            if perturb_flag and is_training:
                # candidate_domains = ZK_PERTURB_DOMAINS - {domain}
                candidate_domains = ["问答", "闲聊", "其他"]
                changed_domain = random.choice(list(candidate_domains))

            text = f"小爱：{changed_domain}\n"
            pre_text += text
            if mask_flag_list[round_cnt] and is_training:
                mask += [1] * len(text)
            else:
                mask += [0] * len(text)
            round_cnt += 1

    text = f"用户：{crt_query}\n小爱："
    pre_text += text
    mask += [0] * len(text)
    post_text = crt_domain + "\n"
    mask += [0] * len(post_text)
    feature = {
        "pre_text": pre_text,
        "post_text": post_text
    }
    if repeat_mask_ratio > 0 or nonrepeat_mask_ratio > 0:
        feature["mask"] = mask
    return feature


def zk_sft_offline_transform(sample, **kwargs):
    is_training = kwargs.get("is_training", True)
    gist_token = kwargs.get("gist_token", None)
    # if gist_token and is_training:
    #     raise AssertionError("should be in inference mode when using gist token currently.")

    only_use_last_round = kwargs.get("only_use_last_round", False)
    pre_text = ZK_FINE_TUNE_PROMPTS["default"]
    if gist_token:
        pre_text = gist_token

    dialogues = sample["messages"]

    if only_use_last_round:
        text = f"用户:{dialogues[-2]['content']}\n"
        pre_text += text
    else:
        for i, dialogue in enumerate(dialogues[:-1]):
            if i % 2 == 0:
                text = f"用户:{dialogue['content']}\n"
            else:
                text = f"小爱:{dialogue['content']}\n"
            pre_text += text

    text = "小爱:"
    pre_text += text

    post_text = dialogues[-1]["content"]
    feature = {
        "pre_text": pre_text,
        "post_text": post_text
    }
    return feature


@add_none_label_flag
def firefly_transform(sample, **kwargs):
    data_format = kwargs["data_format"]
    sample_input = re.sub(r'\n+', '\n', sample["input"]).strip()
    sample_output = re.sub(r'\n+', '\n', sample["target"]).strip()
    if data_format == "glm":
        pre_text = "[Round 1]\n\n问：{}\n\n答：".format(sample_input)
        post_text = "{}\n\n".format(sample_output)
    else:
        pre_text = "{}\n".format(sample_input)
        post_text = "{}\n".format(sample_output)
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


def share_gpt_transform(sample, **kwargs):
    """
    Transform for share-gpt format datasets.

    """
    share_gpt_loss_calc_parts = kwargs.get("share_gpt_loss_calc_part", "all").split(",")

    post_text = ""
    label_flag = []
    for conversation in sample["conversations"]:
        crt_text = f"{conversation['from']}：\n"
        post_text += crt_text
        if share_gpt_loss_calc_parts[0] == "all":
            label_flag += [1] * len(crt_text)
        else:
            label_flag += [0] * len(crt_text)
        crt_text = f"{conversation['value'].strip()}\n"
        post_text += crt_text
        if share_gpt_loss_calc_parts[0] == "all" or conversation["from"] in share_gpt_loss_calc_parts:
            label_flag += [1] * len(crt_text)
        else:
            label_flag += [0] * len(crt_text)
    return {
        "pre_text": "",
        "post_text": post_text,
        "label_flag": label_flag
    }


@add_none_label_flag
def wudao_transform(sample, **kwargs):
    if "concat_samples" in sample:
        sample = sample["concat_samples"]
    else:
        sample = [sample]
    text = ""
    for per_sample in sample:
        text += "%s\n%s\n\n" % (per_sample["title"].strip(), per_sample["content"].strip())
    return {
        "pre_text": "",
        "post_text": text
    }


def codeparrot_transform(sample, **kwargs):
    text = sample["content"]
    return {
        "pre_text": "",
        "post_text": text
    }


@add_none_label_flag
def concat_transform(samples, tokenizer, max_seq_len=1024, concat_strategy="TRUNC"):
    """
    Concat multiple samples to fulfill the max sequence length.

    """
    pre_text_list = samples["pre_text"]
    post_text_list = samples["post_text"]
    label_flag_list = samples.get("label_flag", [None] * len(post_text_list))

    assert concat_strategy in ("TRUNC", "PAD", "TRUNC_OR_PAD", "SPLIT"), \
        f"concat strategy {concat_strategy} is illegal."

    concat_text_list = []
    input_ids_list = []
    label_ids_list = []
    concat_label_flag_list = []

    aggr_text = ""
    input_ids = []
    label_ids = []
    aggr_sample_cnt = 0
    aggr_label_flag = None

    def process_split_concat(aggr_text, input_ids, aggr_tokens, prev_aggr_tok_cnt):
        crt_tokens = aggr_tokens[prev_aggr_tok_cnt:]
        tok_split_first = crt_tokens[: max_seq_len - prev_aggr_tok_cnt]
        aggr_text += tokenizer.decode(tok_split_first)
        concat_text_list.append(aggr_text)
        input_ids += tok_split_first
        input_ids_list.append(input_ids)
        label_ids_list.append(copy.deepcopy(input_ids))

        tok_split_second = crt_tokens[max_seq_len - prev_aggr_tok_cnt:]
        while len(tok_split_second) >= max_seq_len:
            aggr_text = tokenizer.decode(tok_split_second[:max_seq_len])
            concat_text_list.append(aggr_text)
            input_ids = tok_split_second[:max_seq_len]
            input_ids_list.append(input_ids)
            label_ids_list.append(copy.deepcopy(input_ids))
            tok_split_second = tok_split_second[max_seq_len:]
        aggr_text = tokenizer.decode(tok_split_second)
        input_ids = tok_split_second
        aggr_sample_cnt = 1
        return aggr_text, input_ids, aggr_sample_cnt

    def process_trunc_concat(aggr_text, crt_text, input_ids, crt_text_tokens, pre_text_token_len, label_ids,
                             aggr_label_flag=None, crt_label_flag=None):
        aggr_text += crt_text
        concat_text_list.append(aggr_text)

        input_ids += crt_text_tokens
        input_ids_list.append(input_ids)

        label_ids += [-100] * pre_text_token_len + crt_text_tokens[pre_text_token_len:]
        label_ids_list.append(label_ids)

        if crt_label_flag:
            if aggr_label_flag is None:
                aggr_label_flag = crt_label_flag[:]
            else:
                aggr_label_flag += crt_label_flag
        concat_label_flag_list.append(aggr_label_flag)

        aggr_text, input_ids, aggr_sample_cnt, label_ids, aggr_label_flag = "", [], 0, [], None
        return aggr_text, input_ids, aggr_sample_cnt, label_ids, aggr_label_flag

    def process_pad_concat(aggr_text, crt_text, input_ids, crt_text_tokens, pre_text_token_len, label_ids,
                           aggr_label_flag=None, crt_label_flag=None):
        concat_text_list.append(aggr_text)
        input_ids_list.append(input_ids)
        label_ids_list.append(label_ids)
        concat_label_flag_list.append(aggr_label_flag)
        aggr_text, input_ids, aggr_sample_cnt = crt_text, crt_text_tokens, 1
        label_ids = [-100] * pre_text_token_len + crt_text_tokens[pre_text_token_len:]
        aggr_label_flag = crt_label_flag
        return aggr_text, input_ids, aggr_sample_cnt, label_ids, aggr_label_flag

    for pre_text, post_text, label_flag in zip(pre_text_list, post_text_list, label_flag_list):
        if pre_text:
            assert concat_strategy != "SPLIT", f"concat strategy SPLIT is illegal when pre text is not null."
        if label_flag:
            assert concat_strategy != "SPLIT", f"concat strategy SPLIT is illegal when label_flag is provided."

        crt_pre_text_tokens = tokenizer(pre_text)["input_ids"]
        crt_text = pre_text + post_text + tokenizer.eos_token
        crt_all_text_tokens = tokenizer(crt_text)["input_ids"]

        prev_aggr_tok_cnt = len(input_ids)
        crt_tok_cnt = len(crt_all_text_tokens)
        aggr_tok_cnt = prev_aggr_tok_cnt + crt_tok_cnt

        crt_label_flag = None
        if label_flag:
            crt_label_flag = label_flag + [1] * len(tokenizer.eos_token)

        aggr_sample_cnt += 1

        if aggr_tok_cnt >= max_seq_len:
            if concat_strategy == "SPLIT":
                aggr_text, input_ids, aggr_sample_cnt = process_split_concat(
                    aggr_text, input_ids, input_ids + crt_all_text_tokens, prev_aggr_tok_cnt)
            elif concat_strategy == "TRUNC" or aggr_sample_cnt == 1:
                aggr_text, input_ids, aggr_sample_cnt, label_ids, aggr_label_flag = process_trunc_concat(
                    aggr_text, crt_text, input_ids, crt_all_text_tokens, len(crt_pre_text_tokens), label_ids,
                    aggr_label_flag, crt_label_flag)
            elif concat_strategy == "PAD":
                aggr_text, input_ids, aggr_sample_cnt, label_ids, aggr_label_flag = process_pad_concat(
                    aggr_text, crt_text, input_ids, crt_all_text_tokens, len(crt_pre_text_tokens), label_ids,
                    aggr_label_flag, crt_label_flag)
            elif concat_strategy == "TRUNC_OR_PAD":
                if aggr_tok_cnt - max_seq_len <= max_seq_len - prev_aggr_tok_cnt:
                    aggr_text, input_ids, aggr_sample_cnt, label_ids, aggr_label_flag = process_trunc_concat(
                        aggr_text, crt_text, input_ids, crt_all_text_tokens, len(crt_pre_text_tokens), label_ids,
                        aggr_label_flag, crt_label_flag)
                else:
                    aggr_text, input_ids, aggr_sample_cnt, label_ids, aggr_label_flag = process_pad_concat(
                        aggr_text, crt_text, input_ids, crt_all_text_tokens, len(crt_pre_text_tokens), label_ids,
                        aggr_label_flag, crt_label_flag)
        else:
            aggr_text += crt_text
            input_ids += crt_all_text_tokens
            label_ids += [-100] * len(crt_pre_text_tokens) + crt_all_text_tokens[len(crt_pre_text_tokens):]
            if crt_label_flag:
                if aggr_label_flag is None:
                    aggr_label_flag = crt_label_flag[:]
                else:
                    aggr_label_flag += crt_label_flag

    if aggr_text:
        concat_text_list.append(aggr_text)
        input_ids_list.append(input_ids)
        if concat_strategy == "SPLIT":
            label_ids_list.append(copy.deepcopy(input_ids))
        else:
            label_ids_list.append(label_ids)
        concat_label_flag_list.append(aggr_label_flag)

    results = {
        "post_text": concat_text_list
    }
    if concat_label_flag_list[0]:
        results["label_flag"] = concat_label_flag_list
    else:
        results["input_ids"] = input_ids_list
        results["labels"] = label_ids_list
    return results


@change_sample_keywords
@keep_original_sample
@chat_template_wrapper
def general_sft_transform(sample, **kwargs):
    use_think = kwargs.get("use_think", False)
    add_think_tag_in_prompt = kwargs.get("add_think_tag_in_prompt", False)
    think_start_token = kwargs.get("think_start_token", "<think>")
    think_end_token = kwargs.get("think_end_token", "</think>")
    think_prefix_token = kwargs.get("think_prefix_token", "Alright").strip()

    add_think_decision = kwargs.get("add_think_decision", False)
    rewrite_sample_use_think = kwargs.get("rewrite_sample_use_think", 0)   # 0-not rewrite; 1-rewrite True; 2-rewrite False

    sample_use_think = sample.get("sample_use_think", True)
    if rewrite_sample_use_think == 1:
        sample_use_think = True
    elif rewrite_sample_use_think == 2:
        sample_use_think = False

    pre_text = sample["instruction"]
    if "input" in sample:
        pre_text += sample["input"]
    pre_text = pre_text.strip() + "\n"
    post_text = sample["output"]
    if isinstance(post_text, list):
        post_text = post_text[0].strip() + "\n"

    if use_think:
        think_text = sample.get("think", "").strip()
        sample_use_think = sample_use_think and think_text
        if think_text:
            think_text += "\n"

        if add_think_tag_in_prompt:
            pre_text += think_start_token + "\n"
            if sample_use_think and add_think_decision:
                pre_text += think_prefix_token + "\n"
                post_text = think_text + think_end_token + "\n" + post_text
            elif sample_use_think and not add_think_decision:
                post_text = think_prefix_token + "\n" + think_text + think_end_token + "\n" + post_text
            elif not sample_use_think and add_think_decision:
                pre_text += think_end_token
                post_text = "\n" + post_text
            else: # not sample_use_think and not add_think_decision
                post_text = think_end_token + "\n" + post_text
        else:
            assert add_think_decision == False, "`add_think_decision` must be True when `add_think_tag_in_prompt` is False"
            if sample_use_think:
                post_text = (think_start_token + "\n" + think_prefix_token + "\n" +
                             think_text + think_end_token + "\n" + post_text)
            else:
                post_text = think_start_token + think_end_token + "\n" + post_text

    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


def control_sft_transform(sample, **kwargs):
    prompt_template_str = kwargs.get("prompt_template", DEFAULT_CONTROL_SFT_PROMPTS)
    prompt_template = Template(prompt_template_str)

    history = ""
    for dialogue in sample["dialogues"][:-1]:
        parser_codes = dialogue['parser_codes']
        if isinstance(parser_codes, list):
            parser_codes = parser_codes[0]
        query_part = f"Query：\n{dialogue['query']}\n\n"
        user_part = f"User：\n{parser_codes}\n\n"
        assistant_part = f"Assistant：\n{dialogue['skill_codes']}\n\n"
        crt_round_text = f"{query_part}{user_part}{assistant_part}"
        history += crt_round_text

    dialogue = sample["dialogues"][-1]
    query = dialogue["query"]
    context = dialogue.get("context", "")
    pre_text = prompt_template.render(history=history, context=context, query=query)

    post_text = dialogue["parser_codes"]
    if isinstance(post_text, list):
        post_text = post_text[0]
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


def aicreative_sft_transform(sample, **kwargs):
    prompt_template_str = kwargs.get("prompt_template", DEFAULT_CONTROL_SFT_PROMPTS)
    prompt_template = Template(prompt_template_str)

    history = ""
    for dialogue in sample["dialogues"][:-1]:
        parser_codes = dialogue['parser_codes']
        if isinstance(parser_codes, list):
            parser_codes = parser_codes[0]
        query_part = f"Query：\n{dialogue['query']}\n\n"
        user_part = f"User：\n{parser_codes}\n\n"
        assistant_part = f"Assistant：\n{dialogue['skill_codes']}\n\n"
        crt_round_text = f"{query_part}{user_part}{assistant_part}"
        history += crt_round_text

    dialogue = sample["dialogues"][-1]
    query = dialogue["query"]
    context = dialogue.get("context", "")
    pre_text = prompt_template.render(history=history, context=context, query=query)
    pre_text = re.sub("Context:\n\n", "", pre_text)

    post_text = dialogue["parser_codes"]
    if isinstance(post_text, list):
        post_text = post_text[0]
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


def cpt_query_sft_transform(sample, **kwargs):
    prompt_template_str = kwargs.get("prompt_template")
    prompt_template = Template(prompt_template_str, keep_trailing_newline=True)

    history = ""
    for dialogue in sample["dialogues"][:-1]:
        user_part = f"用户：\n{dialogue['query']}\n"
        assistant_part = f"小爱：\n{dialogue['tts'][:30]}\n"
        crt_round_text = f"{user_part}{assistant_part}"
        history += crt_round_text

    dialogue = sample["dialogues"][-1]
    query = dialogue["query"]
    pre_text = prompt_template.render(history=history, query=query)

    post_text = dialogue["parser_codes"]
    if isinstance(post_text, list):
        post_text = post_text[0]
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


@change_sample_keywords
@keep_original_sample
def general_chat_sft_transform(sample, **kwargs):
    tokenizer = kwargs.get("tokenizer")
    use_think = kwargs.get("use_think", True)
    add_think_tag_in_prompt = kwargs.get("add_think_tag_in_prompt", True)
    think_start_token = kwargs.get("think_start_token", "<think>")
    think_end_token = kwargs.get("think_end_token", "</think>")
    pre_text = tokenizer.apply_chat_template(sample["messages"][:-1], tokenize=False, add_generation_prompt=True)
    post_text = sample["messages"][-1]["content"]
    if not use_think:
        if pre_text.strip().endswith(think_start_token):
            pre_text = pre_text.split(think_start_token)[0].strip() + "\n"
        if think_end_token in post_text:
            post_text = post_text.split(think_end_token)[1].strip() + "\n"
    else:
        if add_think_tag_in_prompt:
            if not pre_text.strip().endswith(think_start_token):
                pre_text = pre_text.strip() + "\n" + think_start_token + "\n"
            if post_text.strip().startswith(think_start_token):
                post_text = post_text.split(think_start_token)[-1].strip()
        else:
            if pre_text.strip().endswith(think_start_token):
                pre_text = pre_text.split(think_start_token)[0].strip() + "\n"
            if not post_text.strip().startswith(think_start_token) and think_end_token in post_text:
                post_text = think_start_token + "\n" + post_text.strip() + "\n"
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


@chat_template_wrapper
def general_ability_transform(sample, **kwargs):
    """
    data transform logic for the general ability evaluation data samples.

    """
    general_ability_tag = kwargs.get("general_ability_tag", "")
    template_meta_map = kwargs.get("template_meta_map", {})
    dataset_type = sample["dataset_type"]
    if general_ability_tag:
        dataset_type = f"{general_ability_tag}#{dataset_type}"
    assert dataset_type in template_meta_map
    template_meta_info = template_meta_map[dataset_type]
    eval_method = template_meta_info.get("eval_method", "strict")

    pre_text, post_text = "", ""

    env = Environment()
    env.filters['from_json'] = lambda x: json.loads(x)
    env.filters['eval'] = lambda x: eval(x)

    prompt_prefix = get_general_ability_prefix(sample, template_meta_info, env)

    example_template = Template(template_meta_info["example_template"])
    parsed_results = env.parse(template_meta_info['example_template'])
    referenced_vars = meta.find_undeclared_variables(parsed_results)

    # construct example strings
    examples = sample.get("examples", [])
    example_str_list = []

    for idx, example in enumerate(examples + [sample]):
        example_custom_info = json.loads(example["custom_info"])
        answer = f"{example['label']}\n"
        if eval_method == "match_one" and isinstance(example["label"], list):
            answer = f"{example['label'][0]}\n"
        if 'reason' in example_custom_info and example_custom_info['reason']:
            answer += f"{example_custom_info['reason']}\n"

        if idx == len(examples):
            post_text = answer
            answer = ""

        var_map = {
            "query": example["query"],
            "answer": answer,
        }
        for var in referenced_vars:
            if var in ("query", "answer"):
                continue
            cur_template = env.from_string(template_meta_info[var])
            var_map[var] = cur_template.render(custom_info=example_custom_info)
        example_str = example_template.render(**var_map)
        if idx != len(examples):
            example_str_list.append(example_str)
        else:
            pre_text = example_str

    if example_str_list:
        pre_text = (prompt_prefix +
                    "以下是一些示例\n" + "=" * 10 + "\n" + "\n".join(example_str_list) + "=" * 10 + "\n\n" +
                    "以下是你需要回答的问题\n" + "=" * 10 + "\n" + pre_text)
    else:
        pre_text = prompt_prefix + pre_text

    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


@chat_template_wrapper
def intent_analysis_transform(sample, **kwargs):
    generate_scene = kwargs.get("generate_scene", "normal")
    assert generate_scene in INTENT_ANALYSIS_CONFIG
    instruct_prefix = INTENT_ANALYSIS_CONFIG[generate_scene]["instruct"]

    records = []
    dataset_type = sample["dataset"]
    sample_freq = sample.get("freq", 1)
    dialogues = sample["dialogues"]
    domains = sample["domains"].split(",")
    foreground_apps = sample.get("foreground_apps", "").split(",")
    force_use_online_rst = False
    if isinstance(sample_freq, int):
        force_use_online_rst = len(dialogues) > 2
        sample_freq = [sample_freq] * int(len(dialogues) / 2)
    assert len(sample_freq) == len(dialogues) / 2, "freq list and round list should be the same length."

    round_info_list = []
    query = ""
    for idx, dialogue in enumerate(dialogues):
        if idx % 2 == 0:
            query = dialogue["message"]
        else:
            freq = sample_freq[idx // 2]
            try:
                domain = domains[idx // 2]
            except IndexError:
                logging.warning(f"domain list is shorter than dialogues, use a default domain.")
                logging.warning(f"the sample is {sample}")
                domain = "absentDomain"
            if generate_scene in ("miai_v2") and foreground_apps:
                try:
                    foreground_app = foreground_apps[idx // 2]
                except IndexError:
                    logging.warning(f"foreground_app list is shorter than dialogues, use a default foreground_app.")
                    logging.warning(f"the sample is {sample}")
                    foreground_app = ""
            else:
                foreground_app = ""
            is_agent_recall = "agent" in domain.lower() or "copilot" in domain.lower()
            intent = ""
            reply = dialogue["message"]
            if dataset_type in ("intent_dialogue",):
                fields = dialogue["message"].split("，回复:")
                intent = fields[0]
                reply = ""
                if len(fields) > 1:
                    reply = fields[1]
                slots = None
                code_info = None
                if "，代码:" in intent:
                    # 如果意图中包含代码，则需要解析代码，代码为新增字段
                    intent, code_str = intent.split("，代码:")
                    try:
                        code_info = json.loads(code_str)
                    except json.JSONDecodeError:
                        pass
                if "，槽位:" in intent:
                    intent, slots_str = intent.split("，槽位:")
                    try:
                        slots = json.loads(slots_str)
                    except json.JSONDecodeError:
                        # 如果直接解析失败，尝试将单引号替换为双引号（可能是 Python 字典格式）
                        try:
                            if "'" in slots_str and '"' not in slots_str:
                                slots_str = slots_str.replace("'", '"')
                                slots = json.loads(slots_str)
                            else:
                                raise
                        except (json.JSONDecodeError, Exception) as e:
                            # JSON 解析失败时记录错误并设置 slots 为 None
                            logging.warning(f"Failed to parse slots JSON: {slots_str}, error: {e}")
                            slots = None
                    
                    if slots is not None:
                        if "play_mode" in slots:
                            slots.pop("play_mode")
                        if len(slots_str) >= 100:
                            slots = None
                detail_intent = ""
                if "，意图:" in intent:
                    intent, raw_detail_intent = intent.split("，意图:")
                    if is_agent_recall:
                        try:
                            if not code_info:
                                code_info = json.loads(raw_detail_intent)
                            norm_code = code_info.get("norm_code", "").strip()
                            norm_code = norm_code.replace("\n", "  ")
                            if norm_code:
                                detail_intent = f"功能调用:{norm_code}"
                        except:
                            pass
                    else:
                        detail_intent = f"意图:{raw_detail_intent}"
                if detail_intent:
                    intent += f"，{detail_intent}"
                if not is_agent_recall and slots:
                    intent += "，槽位:" + json.dumps(slots, ensure_ascii=False)
            query_info = query
            add_online_intent = False
            add_foreground_app = False

            if domain != "absentDomain" and generate_scene in ("normal", "miai", "miai_v2", "micar", "soundbox") and intent:
                if query.isdecimal():
                    add_online_intent = False
                elif force_use_online_rst or freq > 5:
                    add_online_intent = True
            if add_online_intent:
                query_info += f"\n参考意图：{intent}"

            if generate_scene in ("miai_v2") and foreground_app:
                if query.isdecimal():
                    add_foreground_app = False
                elif freq > 10:
                    add_foreground_app = True
            if add_foreground_app:
                query_info += f"\n前台应用界面：{foreground_app}"
            round_info_list.append((query_info, reply))
            history_str = ""
            current_str = ""

            if generate_scene in ("normal", "sensitive", "miai", "miai_v2", "micar", "soundbox"):
                if len(round_info_list) > 1:
                    history_str = ""
                    for crt_round_idx, (crt_query_info, crt_reply) in enumerate(round_info_list[:-1]):
                        crt_info_str = f"Round[{crt_round_idx + 1}]\n用户：{crt_query_info}\n小爱：{crt_reply}\n"
                        history_str += crt_info_str
                    history_str = f"历史对话信息：\n{history_str}\n"
                current_str = f"用户当前指令：{round_info_list[-1][0]}\n请分析用户当前指令的意图：\n"
            elif generate_scene in ("phone_guide",):
                for crt_query_info, crt_reply in round_info_list:
                    current_str += f"用户：\n{crt_query_info}\n小爱：\n{crt_reply}\n"
                current_str = f"用户和小爱的对话如下：\n{current_str}\n请给出回答：\n"

            if not (history_str + current_str):
                continue

            pre_text = instruct_prefix + history_str + current_str
            records.append(pre_text)
    return {
        "pre_text": records,
        "post_text": [str(freq) for freq in sample_freq]
    }


@chat_template_wrapper
def rewrite_intent_analysis_transform(sample, **kwargs):
    use_rag = kwargs.get("use_rag", False)
    collection = kwargs.get("rag_vec_db_collection", None)
    emb_model = kwargs.get("rag_emb_model", None)
    n_results = kwargs.get("rag_n_results", 0)
    dist_threshold = kwargs.get("rag_dist_threshold", 0)
    degrade_example_str = kwargs.get("degrade_example_str", None)
    degrade_examples = []
    if degrade_example_str:
        degrade_examples = degrade_example_str.split("<%example_splitter%>")

    dialogues = sample["dialogues"]
    records = []
    for idx in range(len(dialogues)):
        cum_dialogues = dialogues[: idx + 1]
        regen_scene = cum_dialogues[-1].get("regen_scene", "")
        if not regen_scene or regen_scene not in REWRITE_INTENT_ANALYSIS_CONFIG:
            continue
        template_meta_info = REWRITE_INTENT_ANALYSIS_CONFIG[regen_scene]
        prompt_prefix = template_meta_info["instruct"]
        template = Template(template_meta_info["template"])

        retrieved_examples = []
        if use_rag:
            query_for_retrieve = Template(template_meta_info["query_for_retrieve"]).render(dialogues=cum_dialogues)
            retrieved_examples = retrieve(query_for_retrieve, collection, emb_model, n_results, dist_threshold)
            if not retrieved_examples:
                retrieved_examples = degrade_examples

        env = Environment()
        parsed_results = env.parse(template_meta_info['template'])
        referenced_vars = meta.find_undeclared_variables(parsed_results)

        var_map = {}
        for var in referenced_vars:
            cur_template = env.from_string(template_meta_info[var])
            var_map[var] = cur_template.render(dialogues=cum_dialogues, retrieve_results=retrieved_examples)
        record = template.render(**var_map)
        record = prompt_prefix + record
        records.append(record)

    return {
        "pre_text": records,
        "post_text": [""] * len(records)
    }


def multi_choice_transform(sample, **kwargs):

    def construct_question_str(example, output_answer=False):
        crt_example_str = example["question"] + "\n"
        for i, choice in enumerate(example["choices"]):
            crt_example_str += f"{chr(ord('A') + i)}. {choice}\n"
        crt_example_str += "答案：\n"
        if output_answer:
            crt_example_str += chr(ord('A') + example["answer"]) + "\n"
        return crt_example_str

    few_shots_cnt = kwargs.get("few_shots_cnt", 0)
    few_shots_pool = kwargs.get("few_shots_pool", None)
    if few_shots_cnt > 0:
        assert few_shots_pool, "`few_shots_pool` should not be None when `few_shots_cnt` is specified!"
    dataset_subtype = sample["dataset_subtype"]
    dataset_subtype_ch = sample["dataset_subtype_ch"]

    if few_shots_pool is None:
        few_shots_pool = {}

    examples = few_shots_pool.get(dataset_subtype, [])[:few_shots_cnt]
    example_str = ""
    for example in examples:
        crt_example_str = construct_question_str(example, output_answer=True) + "\n"
        example_str += crt_example_str

    target_question_str = construct_question_str(sample, output_answer=False)
    pre_text = (f"以下是关于{dataset_subtype_ch}考试的单项选择题，请选出其中的正确答案。\n\n"
                f"{example_str}{target_question_str}")
    post_text = chr(ord('A') + sample["answer"])
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


@chat_template_wrapper
def instagger_transform(sample, **kwargs):
    pre_text, post_text = "", ""
    for dialog in sample["dialogues"]:
        pre_text += f"{dialog['role']}：{dialog['message']}\n"
    return {
        "pre_text": pre_text,
        "post_text": post_text
    }


@chat_template_wrapper
def complex_episode_judge_transform(sample, **kwargs):
    pre_text, post_text = "", ""
    turn_info_list = []
    for turn in sample["turns"]:
        turn_info = {
            "turn_idx": turn["turn_idx"],
            "query": turn["query"],
            "to_speak": turn["to_speak"],
            "func": turn["func"],
        }
        turn_info_list.append(turn_info)

    template = Template(COMPLEX_EPISODE_JUDGE_PROMPT_TEMPLATE)
    pre_text = template.render(device_category=sample["device_category"],
                               episode_json=json.dumps(turn_info_list, ensure_ascii=False))

    return {
        "pre_text": pre_text,
        "post_text": post_text
    }
