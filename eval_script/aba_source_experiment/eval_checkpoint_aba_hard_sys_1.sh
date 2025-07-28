currentTime=`date "+%m_%d_%H_%M"`
exp_name='sys_1_easy'
export HF_ENDPOINT=https://hf-mirror.com
merge_path="/path/to/your/checkpoint/aba_sys_1_hard05_20_02_39"
#### gsm8k
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task gsm8k --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/ #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/gsm8k/
### math500
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path --task math500 --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/math500/ #  2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/math500/
### AIME 
python ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task aime24 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/aime/  # 2>&1 | tee aime24.txt
python token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/aime/
#### amc
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task amc23 --tp 4 --temperatures 0.7 --n 8 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/amc23/  # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/amc23/
#### minervamath
python  ./skythought/skythought_evals/inference_and_check.py --model $merge_path  --task minervamath --tp 4 --temperatures 0.7 --n 1 --max_tokens 8192 --result-dir ./reason_eval_log/${exp_name}/${currentTime}/minervamath/ --system-prompt-template deepseek-r1 # 2>&1 | tee aime24.txt
python  token_usage_script.py --tokenizer_path $merge_path --path ./reason_eval_log/${exp_name}/${currentTime}/minervamath/