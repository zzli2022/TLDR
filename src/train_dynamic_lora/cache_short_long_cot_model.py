import argparse
import json, os
from vllm import LLM, SamplingParams
import transformers
from math_verify import parse, verify
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
import contextlib
import gc
import torch
import time
import subprocess

def format_prompt_ds(query):
    return "".join((
        "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}.<｜User｜>{query}<｜Assistant｜>".format_map(dict(query=query)),
    ))


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

# sys_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."

def format_prompt(query):
    return "".join((
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "<|im_start|>user\n{query}\n".format_map(dict(query=query)),
        "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "<|im_start|>assistant\n"
    ))


def format_prompt_qwq_preview(query):
    return "".join((
        "<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n",
        "<|im_start|>user\n{query}<|im_end|>\n".format_map(dict(query=query)),
        "<|im_start|>assistant\n",
    ))

    
def save_dict_to_json(data_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    print(f"字典已保存到: {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM inference on eval data")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the JSON config file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model path or name to load in vLLM")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Model path or name to load in vLLM")
    parser.add_argument("--part_num", type=int, required=True,
                        help="Model path or name to load in vLLM")
    parser.add_argument("--output_max_len", type=int, required=True,
                        help="Model path or name to load in vLLM")
    return parser.parse_args()


def main():
    args = parse_args()
    # import pdb; pdb.set_trace()
    # 加载 JSON 配置文件
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # 找到包含 eval_number 的条目
    eval_entry = next((item for item in config if "eval_number" in item), None)
    if eval_entry is None:
        raise ValueError("配置中未找到包含 'eval_number' 的条目")

    eval_data_path = eval_entry["data_path"]
    print(f"读取评估数据自: {eval_data_path}")


    # 加载评估数据
    eval_samples = []
    with open(eval_data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            eval_samples.append(example)

    print(f"共加载 {len(eval_samples)} 条样本")

    # 初始化 vLLM
    print(f"加载模型: {args.model_path}")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.output_max_len)
    llm = LLM(model=args.model_path, enable_sleep_mode=True, tensor_parallel_size=4)
    # 推理并输出
    acc_cnt = 0.0
    token_cnt = 0.0
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=8192,
        padding_side="left",
        use_fast=True,
    )
    times=0
    prompts_list = []
    for i, sample in enumerate(eval_samples):
        # import pdb; pdb.set_trace()
        ## TODO: debug
        if times>args.part_num:
            break
        if os.getenv('FORMAT_PROMPT_TYPE') == "DS":
            prompt = format_prompt_ds(sample.get("query"))
        elif os.getenv('FORMAT_PROMPT_TYPE') == "QwQ_Preview":
            prompt = format_prompt_qwq_preview(sample.get("query"))
        
        prompts_list.append(prompt)
        times+=1

    # import pdb; pdb.set_trace()
    outputs = llm.generate(prompts=prompts_list, sampling_params=sampling_params)
    times = 0
    for prompt, output, sample in zip(prompts_list, outputs, eval_samples):
        if times>args.part_num:
            break
        # print(f"\nSample {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Response: {output.outputs[0].text.strip()}")
        eval_predict_text = output.outputs[0].text.strip()
        gt_answer = sample['ref_answer']
        if verify(parse(extract_answer(eval_predict_text)), parse(gt_answer)):
            acc_cnt += 1
        token_cnt += len(tokenizer.encode(eval_predict_text))
        times+=1
    token_avg = token_cnt/args.part_num
    acc_avg = acc_cnt/args.part_num
    print("Token Avg:{longcot_token_avg}")
    print("Acc Avg:{longcot_acc_avg}")
    cache_dict = {
        "model_name": args.model_path,
        "token":token_avg,
        "acc": acc_avg
    }
    save_dict_to_json(cache_dict, args.output_path)

if __name__ == "__main__":
    main()
