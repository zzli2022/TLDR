currentTime=`date "+%m_%d_%H_%M"`
exp_name='tldr_7b'
export HF_ENDPOINT=https://hf-mirror.com
merge_path="/path/to/your/checkpoint/tldr_system_1_2_data_1_vs_1_max_len_16000"
ref_long_cot_json_file="./devset_cache/DeepSeek-R1-Distill-Qwen-7B-512-16000.json"
python analysis_checkpoint_on_dev.py --eta 0.3 --windows 2 --threshold 0.1 --clip_step checkpoint-32 --input_checkpoint_dir $merge_path --ref_long_cot_json_file $ref_long_cot_json_file
checkpoint_name=$(python analysis_checkpoint_on_dev.py --eta 0.3 --windows 2 --threshold 0.1 --clip_step checkpoint-32 --input_checkpoint_dir $merge_path --ref_long_cot_json_file $ref_long_cot_json_file)
echo "We Select Checkpoint $checkpoint_name"
merge_path=$merge_path/$checkpoint_name
echo "Our Final Checkpoint Path $merge_path"
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