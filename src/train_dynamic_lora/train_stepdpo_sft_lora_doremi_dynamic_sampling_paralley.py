# process data

import json
import math
import random
import copy
import os
import warnings
import pathlib
from typing import Dict, List, Union, Optional, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model
import numpy as np
import deepspeed
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import DistributedSampler
from math_verify import parse, verify
from vllm import LLM, SamplingParams
import torch.distributed as dist
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed, broadcast
import transformers
from transformers import AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer
from trainer_dynamic import Trainer

from extras.vllm_client_new import VLLMClient
from cot_domain import CoTDomain, EvalCoTDomain
from dynamic_callback import Long2ShortDynamicLoadingCallback
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch
from transformers import Conv1D

def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names

import wandb


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_DATA_SEED = 42
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
    'deepseek-r1-14b': (
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}.<｜User｜>{query}<｜Assistant｜>"
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
    'qwq_preview': (
        "<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n"
        "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}. <|im_end|>\n"
        "<|im_start|>assistant\n"
    )
}

import re
import time

def extract_question_parts_qwq_preview(input_text):
    # 使用正则表达式匹配问题描述和最终答案
    question_pattern = r'<\|im_start\|>user(.*?)<\|im_end\|>'
    answer_pattern = r'\\boxed{(\d+(?:\.\d+)?)}'

    # 提取问题和答案
    question_match = re.search(question_pattern, input_text, re.DOTALL)
    answer_match = re.search(answer_pattern, input_text)

    if question_match and answer_match:
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return question, answer
    else:
        return None, None


def extract_question_parts_ds(input_text):
    # 使用正则表达式匹配问题描述和最终答案
    question_pattern = r'<\｜User\｜>(.*?)<\｜Assistant\｜>'
    answer_pattern = r'\\boxed{(\d+(?:\.\d+)?)}'

    # 提取问题和答案
    question_match = re.search(question_pattern, input_text, re.DOTALL)
    answer_match = re.search(answer_pattern, input_text)

    if question_match and answer_match:
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return question, answer
    else:
        return None, None

# def extract_answer(text):
#     match = re.search(r'\\boxed{(.*?)}', text)
#     if match:
#         return match.group(1)  
#     else:
#         return ""  

def extract_answer(string):
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""

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


def check_var(var):
    try:
        var
    except NameError:
        var_exists = False
    else:
        var_exists = True
    return var_exists

def format_prompt_ds(query):
    return "".join((
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}.<｜User｜>{query}<｜Assistant｜>".format_map(dict(query=query)),
    ))

def format_prompt_qwq_preview(query):
    return "".join((
        "<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n",
        "<|im_start|>user\n{query}, Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n".format_map(dict(query=query)),
        "<|im_start|>assistant\n",
    ))

# def format_prompt_qwen(query):
#     return "".join((
#         "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
#         "<|im_start|>user\n{query}\n".format_map(dict(query=query)),
#         "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
#         "<|im_start|>assistant\n"
#     ))

# eval_prompt = 'Please reason step by step, and put your final answer within \\boxed{{}}.\n'
def compute_metrics(pred, is_in_train=True):
    # import pdb; pdb.set_trace()

    inputs_ids = pred.inputs
    labels = pred.label_ids
    metrics = {
        'eval_dev_token': torch.zeros((1), device=tr_accelerator.device)[0]*0.0,
        'eval_dev_acc': torch.zeros((1), device=tr_accelerator.device)[0]*0.0,
    }
    ### init_vllm_client
    if tr_accelerator.is_main_process:
        global vllm_client
        try:
            vllm_client
        except:
            vllm_client = VLLMClient(
                training_args.vllm_server_host, training_args.vllm_server_port, connection_timeout=training_args.vllm_server_timeout
            )
            vllm_client.init_communicator()
    deepspeed_plugin = tr_accelerator.state.deepspeed_plugin
    zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
    gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext

    model.eval()
    torch.cuda.empty_cache()
    if is_peft_model(model):
        with gather_if_zero3(list(model.parameters())):
            # Update vLLM weights while parameters are gathered
            model.merge_adapter()
            for name, param in model.named_parameters():
                if ('lora_B' in name):
                    continue
                if ('lora_A' in name):
                    continue
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                name = name.replace("modules_to_save.default.", "")
                time.sleep(0.2)
                if tr_accelerator.is_main_process:
                    vllm_client.update_named_param(name, param.data)

            # Unmerge adapters while parameters are still gathered
            model.unmerge_adapter()
            # Parameters will automatically be repartitioned when exiting the context
        # tr_accelerator.deepspeed_engine.empty_partition_cache()
    else:
        # For non-PEFT models, simply gather and update each parameter individually.
        for name, param in model.named_parameters():
            with gather_if_zero3([param]):
                if tr_accelerator.is_main_process:
                    time.sleep(0.2)
                    vllm_client.update_named_param(name, param.data)

    torch.cuda.empty_cache()
    # Reset cache on main process
    if tr_accelerator.is_main_process:
        vllm_client.reset_prefix_cache()

    if tr_accelerator.is_main_process:
        print(tokenizer)
        label_texts = []
        pred_texts = []
        label_ans_list = []
        correct_counts = 0
        token_counts = 0
        predict_text_list = []
        problem_input_text_list = []
        gt_answers_list = []
        for problem_input_id in inputs_ids:
            problem_input_id = np.where(problem_input_id == -100, tokenizer.pad_token_id, problem_input_id)
            try:
                problem_input_text = tokenizer.decode(problem_input_id)
            except:
                import pdb; pdb.set_trace()

            try:
                problem_input_text = problem_input_text.replace('<｜end▁of▁sentence｜>', '').replace('<｜begin▁of▁sentence｜>', '')
                if os.getenv('FORMAT_PROMPT_TYPE') == "DS":
                    problem_input_text, gt_answer = extract_question_parts_ds(problem_input_text)
                if os.getenv('FORMAT_PROMPT_TYPE') == "DS":
                    problem_input_text = format_prompt_ds(problem_input_text)
                problem_input_text_list.append(problem_input_text)
                gt_answers_list.append(gt_answer)
            except: 
                continue 

        completion_ids = vllm_client.generate(
                prompts=problem_input_text_list,
                n=training_args.num_generations,
                max_tokens=training_args.max_completion_length,
        )
        eval_predict_text_list = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for eval_predict_text, gt_answer in zip(eval_predict_text_list, gt_answers_list):
            print(eval_predict_text)
            predict_text_list.append(eval_predict_text)
        # import pdb; pdb.set_trace()
            try:
                if verify(parse(extract_answer(eval_predict_text)), parse(gt_answer)):
                    correct_counts += 1
            except:
                continue
            token_counts += len(tokenizer.encode(eval_predict_text))

        metrics.update({
            'eval_dev_token': torch.tensor(token_counts/len(predict_text_list)).cuda(),
            'eval_dev_acc': torch.tensor(correct_counts/len(predict_text_list)).cuda(),
        })
        metrics['eval_dev_token'] = tr_accelerator.reduce(metrics['eval_dev_token'], reduction='sum')
        metrics['eval_dev_acc'] = tr_accelerator.reduce(metrics['eval_dev_acc'], reduction='sum')
    else:
        metrics['eval_dev_token'] = tr_accelerator.reduce(metrics['eval_dev_token'], reduction='sum')
        metrics['eval_dev_acc'] = tr_accelerator.reduce(metrics['eval_dev_acc'], reduction='sum')
     
    model.train()
    return metrics
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/Users/long2short/Desktop/data/huggingface_model/Qwen2.5-0.5B-Instruct")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    skip_tokens: str = field(default=None, metadata={"help": "Path to the training data."})
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
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    data_file_info_path: str = field(default="/Users/long2short/Desktop/code/Dynamic_Long2Short/cot_domain_data_info.json")
    resume_task_prob_file: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    use_lora:  bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    eval_set_path: str = field(default="./eval_set_data/384_samples_6_level_val.jsonl")
    overwrite_output_dir: bool = field(default=True)
    seq_parallel_size: int = field(
        default=1, metadata={"help": "Number of sequences to parallelize"}
    )
    shortcot_model_path: str = '/cpfs/user/long2short/huggingface_model/huggingface_model/Qwen2.5-7B-Instruct'
    # target_token_acc = [(152, 95.0), (554, 97.0)] # target_acc取得short和long之间最大+2, short取mix之间-100, [(352, 95.0), (554, 89.4)]
    target_token_acc: str = '[(152, 95.0), (554, 97.0)]'
    eval_split_ratio: float = field(default=0.05)
    dynamic_update_type: str = field(default='doremi_long2short')
    max_eval_steps: int = field(
        default=32, metadata={"help": "Path to the evaluation data."}
    ) 
    decay: float = field(default=0.5)
    resume_ema_loss_path: Optional[str] = field(default=None)
    template_name: Optional[str] = "qwen25-math-cot"
    set_names: str = field(
        default="short,long", metadata={"help": "Path to the evaluation data."}
    )
    label_names=['labels', 'answer_gts']
    remove_unused_columns= False
    include_inputs_for_metrics= True
    vllm_server_host: str = '0.0.0.0'
    vllm_server_port: int = 30005
    vllm_server_timeout: float = 120
    num_generations: int = 1
    max_completion_length: int = 4096
    eval_data_max_iterations: int = 2
    exp_name: str = 'dynamic_log'
    accelerator_config = {'split_batches': True}
    # accelerator_config = AcceleratorConfig(dispatch_batches=False, )

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

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

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

class DynamicLong2ShortDataset(IterableDataset):  
    def __init__(self, tokenizer, args, local_rank, data_file_info_path: str, dataset_mode: str='train'):
        # training args
        self.args = args
        self.prompt_template = PROMPT_DICT[args.template_name]
        self.local_rank = local_rank
        self.batch_size = args.per_device_train_batch_size
        self.max_length = args.model_max_length
        self.tokenizer = tokenizer
        # is_resume_weight
        self.is_resume_weight = False
        # init cot_domain
        self.data_seed = args.data_seed if args.data_seed is not None and isinstance(args.data_seed, int) else DEFAULT_DATA_SEED
        data_path_list, eval_data_path_list = self.process_data_info(data_file_info_path)  
        self.data_path_list = data_path_list
        self.eval_split_ratio = 0.1
        self.max_iterations = 100000000000000000 if dataset_mode=="train" else args.eval_data_max_iterations
        # init cot_domain
        self.init_eval_domain(eval_data_path_list)
        self.init_cot_domain_name(data_path_list, dataset_mode)
        print(f"In the dataset: data_seed={self.data_seed}")
        ######## Dynamic Loading logic

        self.compute_data_buffer() # data_buffer为啥存在, 感觉还是会存在data_buffer的问题
        self.cot_domain_proportions = []
        self.start_index = 0
        self.max_length = args.model_max_length
        self.cot_domain_sample_index = {cot_domain_id: 0 for cot_domain_id in self.cot_domain_dicts.keys()}
        self.current_proportion = {}
        self.current_eval_dev_metrics = {}
        self.create_dummy_input()
        self.init_proportion()
        self.dataset_mode = dataset_mode

    def create_dummy_input(self):
        if not hasattr(self, "per_total_bsz"):
            # compute num gpus
            world_size = int(os.getenv('WORLD_SIZE'))
            # batch_size
            per_device_train_batch_size = self.args.per_device_train_batch_size
            gradient_accumulation_steps = self.args.gradient_accumulation_steps
            self.per_total_bsz = world_size * per_device_train_batch_size * gradient_accumulation_steps
        
        # per_total_bsz = self.per_total_bsz
        # create dummy input
        # self.dummy_input = list(range(self.max_length))
        print(f"*** local_rank={self.args.local_rank}\t\ttotal_bsz={self.per_total_bsz}")

    def init_proportion(self):
        if not getattr(self, "is_resume_weight", False):
            assert hasattr(self, "cot_domain_prob_list") 
            # 这个proportion是一个整数到概率值的id, 但是buffer有多大要靠这个update_proportion函数来控制
            proportion = {i: item for i, item in enumerate(self.cot_domain_prob_list)} 
            self.current_buffer = self.update_proportion(proportion, 0)
        else:
            print(hasattr(self, "is_resume_weight"))
            assert hasattr(self, "resume_domain_index_list")
            self.current_buffer = self.resume_domain_index_list.copy()
        print(f"after init proportion: current_buffer={len(self.current_buffer)}")

    def init_eval_domain(self, eval_data_path_list: List[str]):
        # import pdb; pdb.set_trace()
        for i, data_path_dict in enumerate(eval_data_path_list):
            self.eval_domain = iter(EvalCoTDomain(data_path_dict['path']))
        # print(f"data_path={data_path_dict['path']}\tmode=eval")

    def init_cot_domain_name(self, data_path_list: List[str], 
                  dataset_mode: str='train'):
        self.cot_domain_dicts = {}
        ## 对于每个CoTDomain, 用一个整数索引, 从一个整数映射到对应的数据, 数据用CoTDomain类包装
        ## 对于每个CoTDomain, 用一个整数索引, 从一个整数映射到对应的cot domain name, 比如"longcot", "shortcot", "...", “...”, xxx
        self.cot_domain_iter_dicts = {}
        self.cot_domain_name = {}
        self.cot_domain2idx = {}
        for i, data_path in enumerate(data_path_list):
            self.cot_domain_name[i] = self.data_path_list[i]
            self.cot_domain2idx[self.data_path_list[i]] = i
            self.cot_domain_dicts[i]: CoTDomain = CoTDomain(data_path, dataset_mode, self.eval_split_ratio)
            print(f"data_path={data_path}\tmode={dataset_mode}")
            print("local_rank={}\tCoTDomain_ids: {}\tCoTDomain Name: {}\tlength={}".format(self.args.local_rank, i, self.data_path_list[i], self.cot_domain_dicts[i].train_data_size))
            print("****************************")
            # build iter
            self.cot_domain_iter_dicts[i] = iter(self.cot_domain_dicts[i])

    def update_proportion(self, new_proportion: Dict[int, float], steps: int):
        # TODO: should pad
        assert isinstance(steps, int) and steps >= 0
        assert hasattr(self, "data_buffer_size") and isinstance(self.data_buffer_size, int)
        #
        print(f"*** global_step: {steps}\t new_proportion: {new_proportion} \t local_rank: {self.args.local_rank}***")

        domain_num_dict = {}
        count = 0
        # import pdb; pdb.set_trace()
        # Check the order of the data
        for domain_idx, weight in new_proportion.items():
            num = int(weight * self.data_buffer_size)
            domain_num_dict[domain_idx] = num
            count += num
        if count < self.data_buffer_size:
            for i in range(self.data_buffer_size - count):
                domain_num_dict[i] += 1
        print(f"*** INFO: new_proportion={new_proportion}\t\tdomain_num_dict={domain_num_dict}\t\tbuffer={self.data_buffer_size}")

        current_buffer = []
        for domain_idx, num in domain_num_dict.items():
            tmp = [domain_idx] * num
            current_buffer.extend(tmp)
        # set random seed
        # print(f"*** global_steps: {steps}\tseed: {self.data_seed + steps}")
        random.seed(self.data_seed + steps)
        random.shuffle(current_buffer)
        print(f"*** global_steps: {steps}\tseed: {self.data_seed + steps}")
        ## NOTE: if steps == 0, then we will add some pad token here
        self.current_buffer = current_buffer
        # print(f"*** local_rank: {self.args.local_rank}\tglobal_step: {steps}\tcurrent_buffer: {len(self.current_buffer)}")
        print(f"*** local_rank: {self.args.local_rank}\tglobal_step: {steps}\tcurrent_buffer: {len(self.current_buffer)}")
        self.current_proportion = new_proportion
        print(f"local_rank={self.local_rank}\tcurrent_proportion={self.current_proportion}")
        # reinit start_index
        self.start_index = 0
        print(f"*** Update: Steps: {steps}\tstart_index={self.start_index}\tlocal_rank={self.local_rank} ***")
        ## update proportions
        cot_domain_weight = [(k, v) for k, v in self.current_proportion.items()]
        cot_domain_weight = sorted(cot_domain_weight, key=lambda x: x[0])
        cot_domain_name = [self.cot_domain_name[k[0]] for k in cot_domain_weight]
        cot_domain_weight = [v[1] for v in cot_domain_weight]
        # update
        infos = {"global_step": steps, "cot_domain_weight": cot_domain_weight, "cot_domain_name": cot_domain_name}
        self.cot_domain_proportions.append(infos)
        return current_buffer
      
    def compute_data_buffer(self):
        # import pdb; pdb.set_trace()
        # compute num gpus
        world_size = int(os.getenv('WORLD_SIZE'))
        # batch_size
        per_device_train_batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        # steps 
        # print(f"eval_steps: {self.args.eval_steps}")
        assert isinstance(self.args.max_eval_steps, (int, float))
        # TODO if eval_steps is float, compute it's number
        eval_steps = self.args.max_eval_steps
        # data_buffer
        per_total_bsz = per_device_train_batch_size * gradient_accumulation_steps
        self.data_buffer_size = int(per_total_bsz * world_size * eval_steps) # 这个来确定是总的buffer size是什么样的
        print(f"*** each buffer size: {self.data_buffer_size}\tworld_size={world_size} ***")

    def process_data(self, example):
        source = self.prompt_template.replace('{query}', 
                                              example['query'])
        target = f"{example['response']}{self.tokenizer.eos_token}"
        return dict(input_ids=source, labels=target)

    def process_eval_data(self, example):
        source = self.prompt_template.replace('{query}', 
                                              example['query'])
        target = 'The answer is \\boxed{' + f"{example['ref_answer']}{'}'}{self.tokenizer.eos_token}"
        return dict(input_ids=source, labels=target)

    def process_data_info(self, data_file_info_path: str):
        if not os.path.exists(data_file_info_path):
            raise FileNotFoundError(f"File {data_file_info_path} is not existed, please check it")
        # check if it is a json file
        _, ext = os.path.splitext(data_file_info_path)
        if ext.lower() != '.json':
            raise ValueError(f"{data_file_info_path} should be a json file")

        with open(data_file_info_path, 'r') as f:
            data_infos = json.load(f)
        f.close()
        print(f"******* data_infos={data_infos} *******")

        assert len(data_infos) > 0, "there is no data info can be used to create datasets!"
        check_name_list = ["data_path", "cot_domain_name", "cot_domain_prob", "eval_number"]
        data_path_list = []
        eval_data_path_list = []
        self.cot_domain_name_list = []
        cot_domain_prob_list = []
        self.cot_domain_prob_list = []
        for data_info in data_infos:
            assert isinstance(data_info, dict)
            # for key in check_name_list:
            #     assert key in data_info
            if "eval_number" in data_info:
                eval_data_path_list.append({'path': data_info["data_path"], 'number': data_info["eval_number"]})
                continue
            # save info
            data_path_list.append(data_info["data_path"]) 
            self.cot_domain_name_list.append(data_info["cot_domain_name"])
            if "cot_domain_prob" in data_info:
                cot_domain_prob_list.append(data_info["cot_domain_prob"])


        print("******* INFO *******")
        print(f"data_path_list: {data_path_list}")
        print(f"cot_domain_name_list: {self.cot_domain_name_list}")
        print(f"cot_domain_prob_list: {cot_domain_prob_list}")
        print(f"eval data path: {eval_data_path_list}")
        # normalize the prob of each task
        self.cot_domain_prob_list = cot_domain_prob_list
        print(f"After normalize: level_prob_list: {self.cot_domain_prob_list}")
        

        return data_path_list, eval_data_path_list

    def __iter__(self):
        world_size = int(os.getenv('WORLD_SIZE'))
        local_rank = self.local_rank
        
        if self.dataset_mode=='eval':
            while True:
                # import pdb; pdb.set_trace()
                if self.max_iterations is not None and self.start_index >= self.max_iterations:
                    self.start_index = 0
                    break
                current_data = next(self.eval_domain)
                curr_process_data = self.process_eval_data(current_data) 
                self.start_index += 1
                yield curr_process_data
        else:
            while True: 
                current_idx = self.start_index % len(self.current_buffer) 
                # assert self.start_index < len(self.current_buffer)
                cot_domain_index = self.current_buffer[current_idx]  
                if cot_domain_index not in self.cot_domain_sample_index and cot_domain_index == -1: 
                    self.cot_domain_sample_index[cot_domain_index] = 0
                self.cot_domain_sample_index[cot_domain_index] += 1 
                # obtain current task's data
                current_data = next(self.cot_domain_iter_dicts[cot_domain_index])

                curr_process_data = self.process_data(current_data) 
                curr_process_data["cot_domain_idx"] = cot_domain_index 
                self.start_index += 1
                yield curr_process_data
           
def create_long2short_callback(data_args, training_args) -> List:
    dynamic_update_type = training_args.dynamic_update_type
    target_token_acc = eval(training_args.target_token_acc)
    set_names = training_args.set_names
    print(target_token_acc)
    dynamic_loading_callback = Long2ShortDynamicLoadingCallback(
        target_token_acc, set_names, dynamic_update_type,
        decay=training_args.decay,
    )
    return [dynamic_loading_callback]

def make_long2short_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    train_dataset = DynamicLong2ShortDataset(tokenizer, training_args, training_args.local_rank, 
                                              data_args.data_file_info_path,
                                              dataset_mode='train')
    eval_dataset = DynamicLong2ShortDataset(tokenizer, training_args, training_args.local_rank, 
                                              data_args.data_file_info_path,
                                              dataset_mode='eval')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

class EvaluateFirstStepCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True

def train():
    global local_rank
    global training_args
    global tr_accelerator
    global trainer
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    data_args.data_length = int(remaining_args[1])
    local_rank = training_args.local_rank
    print(training_args, '==training_args==')
    print(data_args, '==data_args==')
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    try:
        config._attn_implementation = 'flash_attention_2'
    except:
        print('====none====')
    
    global tokenizer
    global model
    global vllm_client

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        config=config
    )
    # import pdb; pdb.set_trace()
    target_list = get_specific_layer_names(model)
    target_set = set(target_list)
    target_set.remove('')
    target_list = list(target_set)
    lora_config = LoraConfig(
        r=data_args.lora_rank,  # LoRA 的秩
        lora_alpha=32,  # LoRA 的 alpha 参数
        target_modules=target_list, #"all-linear", # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块，通常是注意力机制中的 query 和 value 投影
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
    # Load data
    data_module = make_long2short_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    callbacks = create_long2short_callback(data_args, training_args)

    from torch.utils.data import DataLoader 
    # test deocoder pipeline
    test_loader = DataLoader(data_module['train_dataset'],
                              collate_fn=data_module['data_collator'],
                              sampler=None,
                              batch_size=1)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, compute_metrics=compute_metrics, **data_module
    )

    tr_accelerator = trainer.accelerator
    # import pdb; pdb.set_trace()
    if tr_accelerator.is_main_process:
        # import pdb; pdb.set_trace()
        wandb.init(
            project=os.getenv('PROJECT_NAME'),
            config={'exp_name': training_args.exp_name},
            name=training_args.exp_name,
        )

    all_files =  os.listdir(training_args.output_dir)
    ckpt_files = []
    for ckpt in all_files:
        if 'checkpoint' in ckpt:
            ckpt_files.append(os.path.join(training_args.output_dir, ckpt))
    if ckpt_files:
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
    



        
