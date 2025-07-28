import json
import math
import random
import copy
import os
import warnings
import pathlib
import glob
from typing import Dict, List, Union, Optional, Sequence
from dataclasses import dataclass, field
from tqdm import tqdm
from multiprocessing import Pool
import sys
import numpy as np
import time

def deduplicate_by_prompt(dict_list):
    seen_prompts = set()
    unique_dicts = []

    for item in dict_list:
        prompt = item.get('prompt')
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            unique_dicts.append(item)

    return unique_dicts

class EvalCoTDomain:
    def __init__(self,
            file_path: Union[str, None] = None,
            eval_split_ratio  = 0.1
        ):
        self.file_path = file_path
        self.start_idx = 0
        self.data_repo = {} 
        self.load_level_data(self.file_path)

    def load_level_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        cot_domain_data = [json.loads(line.strip()) for line in lines]
        
        self.data_repo = {
            'data': cot_domain_data,  
        }
        self.data_repo['eval'] = cot_domain_data


    def __iter__(self):
        while True: 
            # import pdb; pdb.set_trace()
            cur_start_idx = self.start_idx % len(self.data_repo['data'])  
            cur_data = self.data_repo['data'][cur_start_idx] # 实际上在训练和评估的时候一直使用的是3个样本。所以eval_token才始终是一个值
            self.start_idx += 1 
            yield cur_data  

def my_filter(example, tokenizer, data_args):
    response_len = len(tokenizer(example['response'], 
                add_special_tokens=False)['input_ids'])
    rule1 = response_len >= tokenizer.model_max_length
    rule2 = response_len < data_args.model_min_length
    rule3 = 'show that' in example['query'] or 'prove ' in example['query'] and example['think_process']
    all_wait =  re.findall('(wait)', example['response'].lower())
    all_but =  re.findall('(but)', example['response'].lower())
    all_however =  re.findall('(however)', example['response'].lower())
    all_though =  re.findall('(though)', example['response'].lower())
    all_converse = re.findall('(converse)', example['response'].lower())
    all_complicated = re.findall('(complicated)', example['response'].lower())
    all_approach = re.findall('(another)', example['response'].lower())
    all_different = re.findall('(different)', example['response'].lower())
    all_check = re.findall('(check)', example['response'].lower())
    all_evaluate = re.findall('(evaluate)', example['response'].lower())
    all_try = re.findall('(try)', example['response'].lower())
    all_alternative = re.findall('(alternative)', example['response'].lower())
    
    rule_wait = len(all_wait) > data_args.wait_max_num
    rule_but = len(all_but) > data_args.wait_max_num
    rule_however = len(all_however) > data_args.wait_max_num
    rule_though = len(all_though) > data_args.wait_max_num
    rule_converse = len(all_converse) > data_args.wait_max_num
    rule_complicated = len(all_complicated) > data_args.wait_max_num
    rule_approach = len(all_approach) > data_args.wait_max_num
    rule_different = len(all_different) > data_args.wait_max_num
    rule_check = len(all_check) > data_args.wait_max_num
    rule_evaluate = len(all_evaluate) > data_args.wait_max_num
    rule_try = len(all_try) > data_args.wait_max_num
    
    if rule1 or rule2 or rule3:
        return False
    
    if rule_wait or rule_but or rule_however or rule_though or rule_converse or rule_complicated or rule_approach or rule_different or rule_check or rule_evaluate or rule_try:
        return False

    return True

def my_filter_v1(example, tokenizer, data_args):
    
    # keep all solution-only
    if not example['think_process']:
        return True
    
    response_len = len(tokenizer(example['think_process'], 
                add_special_tokens=False)['input_ids'])
    rule1 = response_len >= tokenizer.model_max_length
    rule2 = response_len < data_args.model_min_length
    rule3 = 'show that' in example['query'] or 'prove ' in example['query'] and example['think_process']
    
    if rule1 or rule2:
        return False
    
    all_wait =  re.findall('(wait)', example['think_process'].lower())
    all_check =  re.findall('(check)', example['think_process'].lower())
    all_but =  re.findall('(but)', example['think_process'].lower())
    all_alternative =  re.findall('(alternative)', example['think_process'].lower())
    
    if 'info' in example:
        info = json.loads(example['info'])
        final_level = info.get('final_level', 'default')
    else:
        final_level = 'default'
    
    if final_level == 'default':
        ratio = 1
    elif int(info['final_level']) < 5:
        ratio = 1
    else:
        ratio = 1
    
    rule1 = len(all_wait) > ratio*wait_dict[final_level]+1
    rule2 = len(all_but) > 2*ratio*wait_dict[final_level]+1
    rule3 = len(all_alternative) > wait_dict[final_level]+1
    rule4 = len(all_check) > ratio*wait_dict[final_level]+1
    if rule1 or rule2 or rule3 or rule4:
        return False
    return True


class CoTDomain:
    def __init__(self,
            file_path: Union[str, None] = None,
            mode: str = "train", 
            eval_split_ratio  = 0.1,
            tokenizer = None,
            args = None
        ):
        self.file_path = file_path
        self.start_idx = 0
        self.data_repo = {} 
        self.mode = mode
        assert self.mode in ["train", "eval"]
        self.eval_split_ratio = eval_split_ratio
        self.load_level_data(self.file_path)
        self.args = args
        self.tokenizer = tokenizer
        if self.args.filter_fn == 'initial':
            self.filter_fn = my_filter
        elif self.args.filter_fn == 'v1':
            self.filter_fn = my_filter_v1
        else:
            self.filter_fn = my_filter
            
        print('====filter_fn====', self.filter_fn)
    def load_level_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        cot_domain_data = [json.loads(line.strip()) for line in lines]
        cot_domain_data = [example for example in cot_domain_data if self.filter_fn(example, self.tokenizer, self.args)==True]
        self.data_repo = {
            'data': cot_domain_data,  # 存储原始数据
        }
        eval_size = int(len(cot_domain_data) * self.eval_split_ratio)
        train_data = cot_domain_data[eval_size:]  # 训练数据（剩余的部分）
        self.train_data_size = len(train_data)
        eval_data = cot_domain_data[:eval_size]   # 评估数据（前 eval_size 个数据）
        eval_data = deduplicate_by_prompt(eval_data)
        self.eval_data_size = len(eval_data)

        
        self.data_repo['train'] = train_data
        self.data_repo['eval'] = eval_data


    def __iter__(self):
        while True: 
            # import pdb; pdb.set_trace()
            cur_start_idx = self.start_idx % len(self.data_repo[self.mode])  
            cur_data = self.data_repo[self.mode][cur_start_idx] # 实际上在训练和评估的时候一直使用的是3个样本。所以eval_token才始终是一个值
            self.start_idx += 1 
            yield cur_data


if __name__ == "__main__":
    start = time.time()
    domain = CoTDomain("/Users/long2short/Desktop/data/gsm8k/liang/prm12k_gsm8k_shortest/GSM8K_PRM12K_static_short_long_mix_cot.short_4_vs_long_1.shortest.json", mode="train")
    end = time.time()
    print(next(iter(level)))
    print(f"Cost time: {end - start}")
