exp_name="DeepSeek-R1-Distill-Qwen-7B-Ties-Dare-Merged-Qwen2.5-Math-7B"
merge_path="/path/to/your/long2short_model_merging/DeepSeek-R1-Distill-Qwen-7B-Ties-Dare-Merged-Qwen2.5-Math-7B"
#### math500
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task math500 --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/math500/ --gpu_memory_utilization 0.95 #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/math500/
#### AIME 
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task aime24 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/aime/ --gpu_memory_utilization 0.95  # 2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/aime/
# #### minervamath
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task minervamath --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/minervamath/ --gpu_memory_utilization 0.95 # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/minervamath/
# #### amc
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task amc23 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/amc23/ --gpu_memory_utilization 0.95 # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/amc23/
#### gsm8k
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task gsm8k --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/ --gpu_memory_utilization 0.95 #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/


exp_name="DeepSeek-R1-Distill-Qwen-14B-Ties-Dare-Merged-Qwen2.5-14B"
merge_path="/path/to/your/long2short_model_merging/DeepSeek-R1-Distill-Qwen-14B-Ties-Dare-Merged-Qwen2.5-14B"
#### math500
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task math500 --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/math500/ --gpu_memory_utilization 0.95 #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/math500/
#### AIME 
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task aime24 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/aime/ --gpu_memory_utilization 0.95  # 2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/aime/
# #### minervamath
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task minervamath --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/minervamath/ --gpu_memory_utilization 0.95 # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/minervamath/
# #### amc
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task amc23 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/amc23/ --gpu_memory_utilization 0.95 # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/amc23/
#### gsm8k
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task gsm8k --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/ --gpu_memory_utilization 0.95 #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/