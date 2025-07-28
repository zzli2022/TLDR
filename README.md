# TL;DR

## 🔧 Training Code

To train the models, you must first deploy an inference acceleration service and record its node IP. Then, run the corresponding training script.

**For DS-R1-7B:**

```bash
# Start the inference accelerator and record the node IP (e.g., 225.13.1.4)
bash ./train_script/train_7B_1_vs_1/7b_parameter_serve.sh

# Begin training using the recorded IP
SERVE_NODE_IP='225.13.1.4' bash ./train_script/train_7B_1_vs_1/max_step_2000_eval_step_32_init_1_vs_1.sh 
```
**For DS-R1-14B:**
```bash
# Start the inference accelerator and record the node IP (e.g., 225.13.1.4)
bash ./train_script/train_14B_1_vs_1/14b_parameter_serve.sh

SERVE_NODE_IP='225.13.1.4' bash ./train_script/train_14B_1_vs_1/max_step_2000_eval_step_32_init_1_vs_1.sh
```

## 📊 Evaluation Code
Evaluation scripts for TLDR and baselines are included. To run evaluation:

```bash
bash ./eval_script/eval_tldr_weight.sh
```

## 📁 Evaluation Results
```bash
bash ./eval_script/eval_tldr_weight.sh
```

## 📦 Dataset
We provide the training data used in our experiments under the ./data/data_repo directory:
- ./data/data_repo/7b_train: Training data for the 7B model  
- ./data/data_repo/14b_data: Training data for the 14B model  
- ./data/data_repo/eval_set: Validation set data
