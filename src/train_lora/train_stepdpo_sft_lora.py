
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from trainer_cp import Trainer as CPTrainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, get_peft_model

from torch.distributed import init_process_group
import datetime
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import argparse
import json
import random

import json, os
import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    # cp train_stepdpo_sft_lora.py train_lora/train_stepdpo_sft_lora.py
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
    'eurus_input': (
        "[INST] "
        "Solve the following math problem step-by-step.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        "[/INST] "
    ),
    "alpaca": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{query}\n\n### Response: Let's think step by step. "
    ),
    'deepseek-r1-14b': (
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}.<｜User｜>{query}<｜Assistant｜>"
    ),
    "qwen2-boxed":(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\nLet's think step by step. "
    ),
    "qwen2-boxed-new":(
        "<|system|>\nYou are a helpful assistant.<|endofsystem|>\n"
        "<|userprompt|>\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|endofuserprompt|>\n"
        "<|response|>\nLet's think step by step. "
    ),
    "deepseek_math": (
    "User: {query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
        ),
    "qwen2-boxed-math": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama2-boxed-math":(
        "[INST] <<SYS>>\nYou are a helpful assistant.<</SYS>>\n\n"
        "{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}. [/INST] "
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama3_1_template": (
    "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    'qwen25_instruct':(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "qwen25-math-cot-control": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{query}\nPlease generate a think_process first with no more than {think_len} tokens, then generate a solution process.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "qwen25-math-cot-control-think": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{query}\nPlease generate a think_process with no more than {think_len} tokens.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    'qwq_preview': (
        "<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n"
        "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}. <|im_end|>\n"
        "<|im_start|>assistant\n"
    )
}
#### 28
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    skip_tokens: str = field(default=None, metadata={"help": "Path to the training data."})
    template_name: Optional[str] = field(default="alpaca")
    trainer_type: Optional[str] = field(default="hf")
    # load_type: Optional[str] = field(default="raw")
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA r"},
    )
    wait_max_num: int = field(
        default=10,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    wait_min_num: int = field(
        default=0,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    model_min_length: int = field(
        default=0,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    filter_fn: str = field(
        default='initial',
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    use_lora:  bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=True)
    seq_parallel_size: int = field(
        default=1, metadata={"help": "Number of sequences to parallelize"}
    )


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

        
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, add_special_tokens=True) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.attention_mask.ne(0).sum().item() for tokenized in tokenized_list
    ] 
    attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
        attention_mask=attention_mask
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    targets_tokenized = _tokenize_fn(targets, tokenizer, add_special_tokens=False)
    
    input_ids = examples_tokenized["input_ids"]
    attention_mask = examples_tokenized['attention_mask']
    return dict(input_ids=input_ids, attention_mask=attention_mask, 
                prompt_len=sources_tokenized['input_ids_lens'],
                answer_len=targets_tokenized['input_ids_lens'])

wait_dict = {'2': 3,
 '3': 3,
 '4': 5,
 '5': 5,
 '6': 10,
 '7': 11,
 '8': 10,
 '9': 11,
 'default': 4}

import re
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

    # if 'THINK_PROCESS' in example['response']:
    #     rule_new = len(tokenizer(example['response'], 
    #                           add_special_tokens=False)['input_ids']) < 256
    #     if rule_new:
    #         return False
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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        from datasets import load_dataset
        from tqdm import tqdm
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        if self.data_args.filter_fn == 'initial':
            filter_fn = my_filter
        elif self.data_args.filter_fn == 'v1':
            filter_fn = my_filter_v1
        else:
            filter_fn = my_filter
            
        print('====filter_fn====', filter_fn)

        # list_data_dict = []
        # print(data_args.data_path.split(','), '===path===')
        # for path in data_args.data_path.split(','):
        #     with open(path) as frobj:
        #         for line in tqdm(frobj):
        #             d = json.loads(line.strip())
        #             flag = my_filter(d, tokenizer, data_args)
        #             if not flag:
        #                 continue
        #             list_data_dict.append(d)

        list_data_dict_ = load_dataset('json',
                          data_files=data_args.data_path.split(','))['train']
        
        self.list_data_dict = list_data_dict_.filter(
            lambda example: filter_fn(example, self.tokenizer, data_args),
            num_proc=32)
        
        print(len(self.list_data_dict), '====after-filter====')

        # print(list_data_dict_, '====after-filter====', tokenizer.model_max_length)

        model_args = kwargs.get('model_args', '')

        # list_data_dict = []
        # for d in list_data_dict_:
        #     list_data_dict.append(d)

#         random.seed(42)
#         list_data_dict = random.sample(list_data_dict,  len(list_data_dict))
#         list_data_dict = list_data_dict[:data_args.data_length]
#         print(list_data_dict[0], '=========list_data_dict=========')

        # logging.warning("Formatting inputs...")
        # print(list_data_dict[0])
        self.prompt_template = PROMPT_DICT[data_args.template_name]
        # import ipdb; ipdb.set_trace()
#         sources = [
#             prompt_template.replace('{query}', example['query'])
#             for example in list_data_dict
#         ]

#         add_bos_token = False
#         # if model_args:
#         #     if 'Llama-3' in model_args.model_name_or_path:
#         #         add_bos_token = True
#         # print(add_bos_token, '==add_bos_token==')
#         # if add_bos_token:
#         #     sources = [f"{tokenizer.bos_token}{example}" for example in sources]

#         targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]

        # self.sources = sources
        # self.targets = targets
        
        # print(len(self.sources), '==sources==')
        # print(self.sources[0], '====', self.targets[0])

    def __len__(self):
        return len(self.list_data_dict)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        example = self.list_data_dict[i]
        source = self.prompt_template.replace('{query}', 
                                              example['query'])
        
        if self.data_args.template_name not in ['qwen25-math-cot-control', 
                                   'qwen25-math-cot-control-think']:
            
            target = f"{example['response']}{self.tokenizer.eos_token}"
        elif self.data_args.template_name == 'qwen25-math-cot-control':
            think_len =len(self.tokenizer(example['think_process'])['input_ids'])
            source = source.replace('{think_len}', think_len)
            
            think_process = f"THINK_PROCESS\n{example['think_process']}\n"
            solution_process = f"SOLUTION_PROCESS\n{example['solution_process']}"
            
            target = f"{think_process}{solution_process}{self.tokenizer.eos_token}"
        elif self.data_args.template_name == 'qwen25-math-cot-control-think':
            think_len =len(self.tokenizer(example['think_process'])['input_ids'])
            source = source.replace('{think_len}', str(think_len))
            target = f"{example['think_process']}{self.tokenizer.eos_token}"
        else:
            target = f"{example['response']}{self.tokenizer.eos_token}"
        return dict(input_ids=source, labels=target)

from transformers import DataCollatorForSeq2Seq

def search(labels, start_id, end_id):
    start_position = []
    end_postion = []
    for idx, label in enumerate(labels):
        if label in start_id:
            start_position.append(idx+1)
        if label in end_id:
            end_postion.append(idx)
    return start_position, end_postion

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class DataCollatorForSupervisedDataset(DataCollatorForSeq2Seq):

    tokenizer: transformers.PreTrainedTokenizer
    skip_token_ids: List[int] = field(default_factory=list)
    
    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        new_batch = {
            'input_ids': data_dict['input_ids'],
            'attention_mask': data_dict['attention_mask']
        }
        label_positions = []
        for prompt_len, answer_len in zip(data_dict['prompt_len'], data_dict['answer_len']):
            label_positions.append((prompt_len, answer_len))
        
        batch = self.tokenizer.pad(
            new_batch,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            pad_to_multiple_of=None,
            return_tensors='pt',
        )
        labels = self._pad_labels(batch["input_ids"], label_positions)
        batch['labels'] = labels
        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, **kwargs) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, **kwargs)
    skip_token_ids = kwargs.get('skip_token_ids', [])
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, skip_token_ids=skip_token_ids)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    data_args.data_length = int(remaining_args[1])
    
    print(training_args, '==training_args==')
    
    print(data_args, '==data_args==')
    
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    try:
        config._attn_implementation = 'flash_attention_2'
    except:
        print('====none====')
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        config=config
    )

    # 定义 LoRA 配置
    lora_config = LoraConfig(
        r=data_args.lora_rank,  # LoRA 的秩
        lora_alpha=32,  # LoRA 的 alpha 参数
        target_modules=["q_proj", "v_proj"],  # 目标模块，通常是注意力机制中的 query 和 value 投影
        lora_dropout=0.1,  # Dropout 概率
        bias="none",  # 是否调整偏置
        task_type="CAUSAL_LM"  # 任务类型
    )

    model.enable_input_require_grads()
    if training_args.use_lora:
        print('=====use lora=====')
        model = get_peft_model(model, lora_config) 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id
     
    if 'tora' in data_args.data_path:
        special_tokens_dict = {'additional_special_tokens': ['<llm-code>', '</llm-code>',
                                                            '<llm-code-output>', '</llm-code-output>']}
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        print('==add-special-tokens==')
        special_token_ids = tokenizer.additional_special_tokens_ids
        skip_token_ids = []
        skip_tokens = set(data_args.skip_tokens.split(','))
        for token, token_id in zip(special_tokens_dict['additional_special_tokens'], special_token_ids):
            if token in skip_tokens:
                skip_token_ids.append(token_id)
    else:
        skip_token_ids = []
        
    if 'qwen2-boxed-new' in data_args.template_name:
        special_tokens_dict = {'additional_special_tokens': 
                               ['<|userprompt|>',
                                '<|endofuserprompt|>',
                                '<|system|>', 
                                '<|endofsystem|>',
                                '<|response|>',
                                '<|endofresponse|>'
                               ]}
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        print('===add special-tokens===')
        print(tokenizer('<|userprompt|>'), '=====')
            

    # if tokenizer.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #         tokenizer=tokenizer,
    #         model=model,
    #     )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, skip_token_ids=skip_token_ids, model_args=model_args)
    
    from torch.utils.data import DataLoader

    test_loader = DataLoader(data_module['train_dataset'],
                              collate_fn=data_module['data_collator'],
                              sampler=None,
                              batch_size=1)
    print('==begin to decode for verification==', len(data_module['train_dataset']))
    for idx, d in enumerate(test_loader):
        input_ids = d['input_ids']
        print(tokenizer.batch_decode(input_ids), '==input==')
        print(d['labels'], '==labels==')
        print(input_ids, '==input_ids==')
        if idx >= 0:
            break
    
    import os
    if data_args.trainer_type == 'hf':
        trainer = Trainer(model=model, tokenizer=tokenizer, 
                          args=training_args, **data_module)
    else:
        trainer = CPTrainer(model=model, tokenizer=tokenizer, 
                          args=training_args, **data_module)
    
    # all_files =  os.listdir(training_args.output_dir)
    ckpt_files = []
    # for ckpt in all_files:
    #     if 'checkpoint' in ckpt:
    #         ckpt_files.append(os.path.join(training_args.output_dir, ckpt))
    if ckpt_files:
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False
    # trainer.add_callback(SavePeftModelCallback)  ## Debug
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir) ## Debug

if __name__ == "__main__":
    train()
