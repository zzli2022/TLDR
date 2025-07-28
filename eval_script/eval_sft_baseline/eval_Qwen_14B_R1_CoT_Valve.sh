currentTime=`date "+%m_%d_%H_%M"`
exp_name='sft_baseline_Qwen_14B_R1_lora_2_CoT_Valve_5_Epoch'
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT=https://hf-mirror.com
exp_name='sft_baseline_QwQ_Preview_14B_CoT_Valve_5_Epoch'
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ENDPOINT=https://hf-mirror.com
lora_path="/path/to/your/checkpoint//Qwen_R1_14B_R1_lora_2_MixChain_C_GSM8K_5_Epoch05_15_21_23"
merge_path="/path/to/your/merged_checkpoint/"$currentTime
python merge_lora.py --base_model /cpfs/user/long2short/huggingface_model/huggingface_model/DeepSeek-R1-Distill-Qwen-14B --lora_model $lora_path --output_dir $merge_path &&  echo $lora_path > $merge_path/created_file.txt
#### gsm8k
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task gsm8k --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/ #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/
#### math500
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task math500 --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/math500/ #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/math500/
# ### AIME 
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task aime24 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/aime/  # 2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/aime/
# #### amc
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task amc23 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/amc23/  # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/amc23/
# # #### minervamath
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task minervamath --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/minervamath/ --system-prompt-template deepseek-r1 # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/minervamath/