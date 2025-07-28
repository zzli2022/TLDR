export WANDB_API_KEY="your_wandb_key"
currentTime=`date "+%m_%d_%H_%M"`
#!/bin/bash
N_GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
N_GPUS=4
echo "Number of GPUS: $N_GPUS"
# WORLD_SIZE=4
NUM_PROCESSES=$(expr $N_GPUS \* $WORLD_SIZE)
# NUM_PROCESSES=1
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
 
export UPDATE_TAG=1
export FORMAT_PROMPT_TYPE="DS"
export MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B # QwQ-32B-Preview
export DS_CONFIG_PATH=./src/config/zero3_config_no_offload.json
export OUTPUT_PATH=/path/to/your/exp_output
 
currentTime=`date "+%m_%d_%H_%M"`
exp_name="aba_sys_1_medium"
export OUTPUT_PATH=$OUTPUT_PATH$exp_name$currentTime

export MODEL_DIR=/cpfs/user/long2short/huggingface_model/huggingface_model
export DATA_DIR=/path/to/your/hf_datasets
 
export PROJECT_NAME=r1_7b_new_gsm8k_data_1_vs_1
export ETA=4
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p ${OUTPUT_PATH}
fi
NCCL_SOCKET_IFNAME=eth0

PART_NUM=512
MODEL_MAX_LENGTH=16000
SHORTCOT_MODEL_NAME=Qwen2.5-7B-Instruct
DATA_FILE_INFO_PATH=./data/system_1_2_config/7b_data/cot_domain_aba_7b_sys_1_medium_data.json  # 用1.5B的低质量SFT刷的, 建议重新跑
### preload_dev_set and inference longcot and shortcot into cache dir
if [ -f "devset_cache/$SHORTCOT_MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json" ]; then
    echo "File Exit: 'devset_cache/$SHORTCOT_MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json'"
    # 提取 token 和 acc 的值
    short_token=$(jq '.token' "devset_cache/$SHORTCOT_MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")
    short_acc=$(jq '.acc' "devset_cache/$SHORTCOT_MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")

    # 保留两位小数
    short_token=$(printf "%.2f" "$short_token")
    short_acc=$(printf "%.2f" "$short_acc")
    echo "token: $short_token"
    echo "acc: $short_acc"
else
    echo "File does not exist. Executing operation."
    CUDA_VISIBLE_DEVICES=4,5,6,7 python src/train_dynamic_lora/cache_short_long_cot_model.py \
      --config_path ${DATA_FILE_INFO_PATH} \
      --model_path ${MODEL_DIR}/${SHORTCOT_MODEL_NAME}/ \
      --part_num $PART_NUM \
      --output_max_len $MODEL_MAX_LENGTH \
      --output_path devset_cache/$SHORTCOT_MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json 
    echo "Execution complete."
    # 提取 token 和 acc 的值
    short_token=$(jq '.token' "devset_cache/$SHORTCOT_MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")
    short_acc=$(jq '.acc' "devset_cache/$SHORTCOT_MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")

    short_token=$(printf "%.2f" "$short_token")
    short_acc=$(printf "%.2f" "$short_acc")
    echo "token: $short_token"
    echo "acc: $short_acc"
fi

if [ -f "devset_cache/$MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json" ]; then
    echo "File Exit: "devset_cache/$MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH""
    # 提取 token 和 acc 的值
    long_token=$(jq '.token' "devset_cache/$MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")
    long_acc=$(jq '.acc' "devset_cache/$MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")

    long_token=$(printf "%.2f" "$long_token")
    long_acc=$(printf "%.2f" "$long_acc")
    echo "token: $long_token"
    echo "acc: $long_acc"
else
    echo "File does not exist. Executing operation."
    CUDA_VISIBLE_DEVICES=4,5,6,7 python src/train_dynamic_lora/cache_short_long_cot_model.py \
      --config_path ${DATA_FILE_INFO_PATH} \
      --model_path ${MODEL_DIR}/${MODEL_NAME}/ \
      --part_num $PART_NUM \
      --output_max_len $MODEL_MAX_LENGTH \
      --output_path devset_cache/$MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json 
    echo "Finish"
    # 提取 token 和 acc 的值
    long_token=$(jq '.token' "devset_cache/$MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")
    long_acc=$(jq '.acc' "devset_cache/$MODEL_NAME-$PART_NUM-$MODEL_MAX_LENGTH.json")

    long_token=$(printf "%.2f" "$long_token")
    long_acc=$(printf "%.2f" "$long_acc")
    echo "token: $long_token"
    echo "acc: $long_acc"
fi

echo "[($short_token, $short_acc), ($long_token, $long_acc)]"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  accelerate launch \
    --main_process_ip="${MASTER_ADDR}" \
    --main_process_port="63340" \
    --machine_rank="0" \
    --num_processes="8" \
    --num_machines="1" \
    src/train_dynamic_lora/train_stepdpo_sft_lora_doremi_dynamic_sampling_paralley.py \
    --do_train \
    --wait_max_num 11 \
    --use_lora False \
    --shortcot_model_path /path/to/your/huggingface_model/Qwen2.5-7B-Instruct \
    --ddp_timeout=10800 \
    --lora_rank 4 \
    --lora_alpha 32 \
    --lora_nums 4 \
    --bf16 \
    --lora_dropout 0.1 \
    --exp_name aba_sys_1_medium \
    --template_name 'deepseek-r1-14b' \
    --data_length 1000000000000 \
    --model_name_or_path ${MODEL_DIR}/${MODEL_NAME}/ \
    --data_file_info_path ${DATA_FILE_INFO_PATH} \
    --output_dir ${OUTPUT_PATH} \
    --max_steps 2000 \
    --model_max_length $MODEL_MAX_LENGTH \
    --include_for_metrics inputs \
    --eval_do_concat_batches \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size $PART_NUM \
    --eval_data_max_iterations $PART_NUM \
    --max_completion_length 8192 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --eval_split_ratio 0.1 \
    --evaluation_strategy "steps" \
    --eval_steps 32 \
    --save_strategy "steps" \
    --save_steps 32 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --split_batches True \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG_PATH} \
    --vllm_server_host 22.4.81.179 \
    --vllm_server_port 30008 \
    --vllm_server_timeout 120 \
    --target_token_acc "[($short_token, $short_acc), ($long_token, $long_acc)]"