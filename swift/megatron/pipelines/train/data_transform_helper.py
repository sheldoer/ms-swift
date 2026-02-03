#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Author wangsiwen@xiaomi.com
    @file data_transform_helper.py
    @Date 2024/11/13 下午8:12
    @Describe 
    @Version 1.0
"""

import json
from functools import wraps

from jinja2 import meta
from .data_transform_config import ROLE_NAME_MAP


import json
from functools import wraps
from collections.abc import Mapping, Sequence


def _to_jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    try:
        import pyarrow as pa
        if isinstance(obj, pa.Scalar):
            return obj.as_py()
    except Exception:
        pass

    if isinstance(obj, Mapping) or hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        return {k: _to_jsonable(obj[k]) for k in obj.keys()}

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_jsonable(x) for x in obj]

    return str(obj)


def chat_template_wrapper(func):
    """
    A decorator for processing output for the ease of using the chat format.

    """
    @wraps(func)
    def wrap_chat_template(sample, **kwargs):
        result = func(sample, **kwargs)
        pre_text = result["pre_text"]
        post_text = result["post_text"]
        use_chat_template = kwargs.get("use_chat_template", False)
        tokenizer = kwargs.get("tokenizer", None)
        model_type = kwargs.get("model_type", None)

        is_list = isinstance(pre_text, list)
        if use_chat_template:
            assert tokenizer, "tokenizer should be specified when using chat template."
            user_role_name, _ = ROLE_NAME_MAP.get(model_type, ("user", "assistant"))
            if is_list:
                for i in range(len(pre_text)):
                    messages = [
                        {"role": user_role_name, "content": pre_text[i]}
                    ]
                    pre_text[i] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                messages = [
                    {"role": user_role_name, "content": pre_text}
                ]
                pre_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        new_result = {"pre_text": pre_text, "post_text": post_text}
        for key in result:
            if key not in new_result:
                new_result[key] = result[key]
        return new_result

    return wrap_chat_template


def change_sample_keywords(func):
    """
    The decorator to change the keys of the samples.

    """
    @wraps(func)
    def change_sample_keywords_func(sample, **kwargs):
        change_keywords_type = kwargs.get("change_keywords_type", "")
        result = func(sample, **kwargs)
        original_keys = result.keys()

        keyword_map = {"pre_text": "pre_text", "post_text": "post_text"}
        if change_keywords_type == "grpo":
            keyword_map = {"pre_text": "prompt", "post_text": "ground_truth"}

        new_result = {}
        for old_key, new_key in keyword_map.items():
            new_result[new_key] = result[old_key]
        for key in original_keys:
            if key not in keyword_map:
                new_result[key] = result[key]
        return new_result

    return change_sample_keywords_func


def keep_original_sample(func):
    """
    The decorator to keep the original sample.

    """
    @wraps(func)
    def keep_original_sample_func(sample, **kwargs):
        keep_raw_sample = kwargs.get("keep_raw_sample", False)
        result = func(sample, **kwargs)
        if keep_raw_sample:
            materialized = _to_jsonable(sample)
            if isinstance(materialized, dict) and all(isinstance(v, list) for v in materialized.values()):
                # dict of lists -> list of dicts
                n = len(next(iter(materialized.values()), []))
                rows = [{k: materialized[k][i] for k in materialized} for i in range(n)]
                result["raw_sample"] = [json.dumps(_to_jsonable(r), ensure_ascii=False) for r in rows]
            else:
                result["raw_sample"] = json.dumps(materialized, ensure_ascii=False)

        return result

    return keep_original_sample_func


def add_none_label_flag(func):
    """
    The decorator the added a none label flag in samples for the consistency of mixed datasets.

    """
    @wraps(func)
    def add_none_label_flag_func(sample, **kwargs):
        result = func(sample, **kwargs)
        if "label_flag" not in result:
            result["label_flag"] = None
        return result

    return add_none_label_flag_func


def retrieve(query, collection, emb_model, n_results, dist_threshold):
    embeddings = emb_model.encode_queries([query])
    raw_results = collection.query(query_embeddings=embeddings, n_results=n_results)
    distances = raw_results["distances"][0]
    documents = raw_results["documents"][0]
    ret_text_list = []
    for dist, doc in zip(distances, documents):
        if dist < dist_threshold:
            ret_text_list.append(json.loads(doc)["content"])
    return ret_text_list


def get_general_ability_prefix(sample, meta_info, env):
    prompt_prefix = meta_info["instruct"]
    prefix_parsed_results = env.parse(prompt_prefix)
    prefix_referenced_vars = meta.find_undeclared_variables(prefix_parsed_results)
    if prefix_referenced_vars:
        var_map = {}
        instruct_info = {}
        if "candidate_func_list" in prefix_referenced_vars:
            label_list = [sample["label"]]
            for example in sample.get("examples", []):
                label_list.append(example["label"])
            instruct_info["candidate_func_list"] = get_candidate_func_list(label_list, meta_info)
        for var in prefix_referenced_vars:
            cur_template = env.from_string(meta_info[var])
            var_map[var] = cur_template.render(instruct_info=instruct_info)
        prompt_prefix = env.from_string(prompt_prefix).render(**var_map)
    return prompt_prefix


def get_candidate_func_list(label_list, meta_info):
    candidate_func_list = meta_info.get("required_func_list", [])[:]
    optional_func_map = {func["name"]: func for func in meta_info.get("optional_func_list", [])}
    extra_func_to_add = meta_info.get("extra_func_to_add", {})

    current_func_name_set = {func["name"] for func in candidate_func_list}
    for label in label_list:
        code_list = label.strip().split("\n")
        for code in code_list:
            crt_func_name = code.split("(")[0]
            for func_name in [crt_func_name] + extra_func_to_add.get(crt_func_name, []):
                if func_name in current_func_name_set:
                    continue
                if func_name in optional_func_map:
                    candidate_func_list.append(optional_func_map[func_name])
                    current_func_name_set.add(func_name)
    return candidate_func_list



