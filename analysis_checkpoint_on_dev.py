import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Find best checkpoint by dev token accuracy.")
    parser.add_argument('--input_checkpoint_dir', type=str, required=True, help="Directory containing checkpoint-* folders")
    parser.add_argument('--ref_long_cot_json_file', type=str, required=True, help="Directory containing checkpoint-* folders")
    parser.add_argument('--eta', type=float, required=True, help="Directory containing checkpoint-* folders")
    parser.add_argument('--windows', type=int, required=True, help="Directory containing checkpoint-* folders") # 
    parser.add_argument('--clip_step', type=str, required=True, help="Directory containing checkpoint-* folders")
    parser.add_argument('--threshold', type=float, required=True, help="Directory containing checkpoint-* folders")
    return parser.parse_args()

import numpy as np


def sort_checkpoint_keys(keys):
    return sorted(keys, key=lambda x: int(x.split('-')[1]))

def find_plateau_indices(arr, window_size=5, threshold=0.01):
    arr = np.array(arr)
    indices = []

    for i in range(len(arr) - window_size + 1):
        window = arr[i:i + window_size]
        if np.max(window) - np.min(window) <= threshold:
            indices.append(i)

    return indices

def filter_checkpoints_by_step(ckpt_dict, max_ckpt_name):
    # import pdb; pdb.set_trace()
    def extract_step(ckpt_name):
        return int(ckpt_name.split("-")[1])
    
    max_step = extract_step(max_ckpt_name)
    output = {k: v for k, v in ckpt_dict.items() if extract_step(k) <= max_step}
    return {
        k: v for k, v in ckpt_dict.items()
        if extract_step(k) >= max_step
    }

def find_min_token_checkpoint(data: dict, ref_acc: float):
    filtered = {
        name: info
        for name, info in data.items()
        if info['eval_dev_acc'] > ref_acc
    }
    if not filtered:
        return None  # 没有满足条件的项
    # 找出eval_dev_token最小的那个
    min_checkpoint = min(
        filtered.items(),
        key=lambda item: item[1]['eval_dev_token']
    )
    return min_checkpoint  

def search_best_checkpoint(args, ckpt_dev_token_acc_dict, ckpt_proportion_dict, ref_long_cot_data):
    # import pdb; pdb.set_trace()
    eta = args.eta
    # import pdb; pdb.set_trace()
    ckpt_proportion_list = list(ckpt_proportion_dict.values())
    ckpt_proportion_list = [data['cot_domain_weight'][1] for data in ckpt_proportion_list[0]]
    indices_2_checkpoint_step = {indice:checkpoint_step for indice, checkpoint_step in enumerate(sort_checkpoint_keys(ckpt_proportion_dict.keys()))}
    stable_proportion_indice = find_plateau_indices(ckpt_proportion_list, window_size=args.windows, threshold=args.threshold)[0]
    stable_proportion_step = indices_2_checkpoint_step[stable_proportion_indice]
    clip_step = args.clip_step
    ckpt_dev_token_acc_dict = filter_checkpoints_by_step(ckpt_dev_token_acc_dict, clip_step)
    ref_acc = ref_long_cot_data["acc"]*(1-eta)
    # ref_token = ref_long_cot_data["token"]
    ref_checkpoint_name = find_min_token_checkpoint(ckpt_dev_token_acc_dict, ref_acc)[0]
    print(ref_checkpoint_name)

def read_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f) 
            return data
    except Exception as e:
        print(f"Warning: Failed to read {json_path}: {e}")
        return None

def main():
    args = parse_args()
    base_dir = args.input_checkpoint_dir
    ref_long_cot_data = read_json(args.ref_long_cot_json_file)
    checkpoint_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("checkpoint-")
    ]

    if not checkpoint_dirs:
        print("No checkpoint-* directories found.")
        return

    ckpt_name_to_metrics_dict = {}
    ckpt_proportion_dict = {}
    for ckpt_dir in checkpoint_dirs:
        eval_dev_token_acc_path = os.path.join(ckpt_dir, "eval_dev_token_acc.json")
        proportion_path = os.path.join(ckpt_dir, "long2short_proportions.json")
        ckpt_dev_metrics = read_json(eval_dev_token_acc_path)
        ckpt_dev_proportion = read_json(proportion_path)
        ckpt_name = os.path.basename(ckpt_dir)
        ckpt_name_to_metrics_dict[ckpt_name] = ckpt_dev_metrics
        ckpt_proportion_dict[ckpt_name] = ckpt_dev_proportion
    search_best_checkpoint(args, ckpt_name_to_metrics_dict, ckpt_proportion_dict, ref_long_cot_data)

if __name__ == "__main__":
    main()
