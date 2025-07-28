import os
import json
import argparse
import transformers
def main(args):
    path = args.path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        padding_side="left",
        use_fast=True
    )
    total_tokens = 0
    count = 0
    correct_count = 0
    total_responses = 0

    for filename in os.listdir(path):
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for key, value in data.items():
                    # import pdb; pdb.set_trace()
                    if "responses" in value:
                        for temp, response_list in value["responses"].items():
                            for response in response_list:
                                if "content" in response:
                                    if response["content"]:
                                        total_tokens += len(tokenizer(response["content"], 
                                                    add_special_tokens=False)['input_ids'])

                    # response_len = len(tokenizer(value['response'], 
                    #             add_special_tokens=False)['input_ids'])
                    # 统计 completion_tokens
                    if "token_usages" in value:
                        for temp, tokens_list in value["token_usages"].items():
                            for tokens in tokens_list:
                                if "completion_tokens" in tokens:
                                    # total_tokens += tokens["completion_tokens"]
                                    count += 1
                    
                    # 统计 correctness
                    if "responses" in value:
                        for temp, response_list in value["responses"].items():
                            for response in response_list:
                                if "correctness" in response:
                                    total_responses += 1
                                    if response["correctness"]:
                                        correct_count += 1

    # 计算平均值和准确率
    average_tokens = total_tokens / count if count > 0 else 0
    accuracy = correct_count / total_responses if total_responses > 0 else 0

    # 输出结果到文件
    output_file = os.path.join(path, "output_token.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total completion tokens: {total_tokens}\n")
        f.write(f"Average completion tokens: {average_tokens}\n")
        print(f"Average completion tokens: {average_tokens}\n")
        f.write(f"Total responses: {total_responses}\n")
        f.write(f"Correct responses: {correct_count}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        print(f"Accuracy: {accuracy:.4f}\n")

    print(f"Result Output To: {output_file}")

if __name__ == "__main__":
    # 设置 argparse
    parser = argparse.ArgumentParser(description="input json path")
    parser.add_argument("--path", type=str, help="dir of input json path")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="tokenizer path"
    )
    args = parser.parse_args()
    main(args)