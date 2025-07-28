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
            cur_start_idx = self.start_idx % len(self.data_repo['data'])  
            cur_data = self.data_repo['data'][cur_start_idx] 
            self.start_idx += 1 
            yield cur_data  

class CoTDomain:
    def __init__(self,
            file_path: Union[str, None] = None,
            mode: str = "train", 
            eval_split_ratio  = 0.1
        ):
        self.file_path = file_path
        self.start_idx = 0
        self.data_repo = {} 
        self.mode = mode
        assert self.mode in ["train", "eval"]
        self.eval_split_ratio = eval_split_ratio
        self.load_level_data(self.file_path)

    def load_level_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        cot_domain_data = [json.loads(line.strip()) for line in lines]
        
        self.data_repo = {
            'data': cot_domain_data,  # 存储原始数据
        }
        eval_size = int(len(cot_domain_data) * self.eval_split_ratio)
        train_data = cot_domain_data[eval_size:] 
        self.train_data_size = len(train_data)
        eval_data = cot_domain_data[:eval_size]  
        eval_data = deduplicate_by_prompt(eval_data)
        self.eval_data_size = len(eval_data)

        
        self.data_repo['train'] = train_data
        self.data_repo['eval'] = eval_data


    def __iter__(self):
        while True: 
            # import pdb; pdb.set_trace()
            cur_start_idx = self.start_idx % len(self.data_repo[self.mode])  
            cur_data = self.data_repo[self.mode][cur_start_idx] 
            self.start_idx += 1 
            yield cur_data


if __name__ == "__main__":
    start = time.time()
    end = time.time()
    # print(next(iter(level)))
    print(f"Cost time: {end - start}")
