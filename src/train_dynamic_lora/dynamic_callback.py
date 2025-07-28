from pathlib import Path
import os
import json
import math
from typing import Dict, List, Union
from functools import partial
import numpy as np
import torch

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.integrations.integration_utils import TensorBoardCallback, WandbCallback
from transformers import TrainerCallback

logger = logging.get_logger(__name__)
import wandb

class Long2ShortDynamicLoadingCallback(TrainerCallback): # TrainerCallbackHandler
    def __init__(self,
                 target_token_acc: List[str] = None,
                 set_names: List[str] = None,
                 update_type: str="long2short",
                 decay: float=0.5,
                 ):
        self.set_names = set_names
        self.cot_domains = set_names.split(",") # "level1, level2, level3, level4, level5"
        self.update_type = update_type 
        self.process_target_token_acc(target_token_acc) # ref model loss
        self.update_cot_domain_num = 0
        self.current_token_acc = {}
        print("Target Token:", self.target_token)
        print("Target Acc:", self.target_acc)
        # assertion
        assert len(self.target_token) == len(self.cot_domains)
        print(f"*** In the Callback: target_tokens={self.target_token} ***")
    
    def process_target_token_acc(self, target_token_acc: List[tuple]):
        assert len(target_token_acc) > 0
        num = len(target_token_acc[0])
        # 分别存储两个domain的token和acc
        self.target_token = []
        self.target_acc   = []

        for i, data_tuple in enumerate(target_token_acc):
            assert len(data_tuple) == num
        
            if num == 2:
                self.target_token.append(data_tuple[0])
                self.target_acc.append(data_tuple[1])
            else:
                raise ValueError("data_tuple size is 2")

    def compute_benifit_for_longcot(self, cur_acc, ref_long_acc, ref_short_acc):
        longcot_benifit = max((ref_long_acc-cur_acc)/abs(ref_long_acc-ref_short_acc), 0)
        return longcot_benifit

    def compute_benifit_for_shortcot(self, cur_token, ref_long_token, ref_short_token):
        shortcot_benifit = max((cur_token-ref_short_token)/abs(ref_long_token-ref_short_token), 0)
        return shortcot_benifit

    def update_proportion(self, current_prop, cur_cot_domain_benifits):
        """ Update the proportion of each domain """
        print(f"*** for update proportion: current_prop: {current_prop}\t")
        eta = int(os.getenv('ETA', 4))
        c = 1e-4 # following Doremi (Xie et al., 2023)
        diff = torch.tensor(cur_cot_domain_benifits)
        # import pdb; pdb.set_trace()
        if self.update_type == "doremi_long2short": # update with exponential descent
            updated_alpha = torch.log(torch.tensor(current_prop)) + eta * diff 
            print(f"after log: update_alpha: {updated_alpha}")
            updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
            print(f"after softmax: updated_alpha: {updated_alpha}")
            updated_domain_weights = (1-c) * updated_alpha + c / len(self.cot_domains)
        elif self.update_type == "constant": # constant proportion
            updated_domain_weights = torch.tensor(current_prop)
    
        updated_domain_weights = updated_domain_weights.numpy().astype('float64')
        updated_domain_weights = updated_domain_weights / updated_domain_weights.sum()
        return updated_domain_weights.tolist()

    def trans_dict_to_list(self, target_dict: Dict):
        target_list = [(k, v) for k, v in target_dict.items()]
        target_list = sorted(target_list, key=lambda x: x[0])
        
        return [v[1] for v in target_list]

    def on_evaluate(self, args, state, control, **kwargs): # on_step_begin()
        # import pdb; pdb.set_trace()
        """ Update the proportion of each domain after each evaluation and update the dataset """
        metrics = kwargs["metrics"]
        # get target longcot acc and longcot token
        k_names = ['eval_shortcot_token', 'eval_shortcot_acc', 'eval_longcot_token', 'eval_longcot_acc']
        if ('eval_shortcot_token' in metrics) and ('eval_longcot_acc' in metrics):
            self.target_acc[0] = metrics['eval_shortcot_acc']
            self.target_token[0] = metrics['eval_shortcot_token']
            self.target_acc[1] = metrics['eval_longcot_acc']
            self.target_token[1] = metrics['eval_longcot_token']
            return 
        # get variables
        logger.info(f"*** local_rank={args.local_rank}\ton evaluate ***")
        train_dataloader = kwargs["train_dataloader"]
        eval_dataloader = kwargs["eval_dataloader"]
        current_prop = train_dataloader.dataset.current_proportion # task_id: proportion
        idx2name = train_dataloader.dataset.cot_domain_name # domain_idx: domain list
        self.cot_domain2idx = train_dataloader.dataset.cot_domain2idx # dict: domain name to number
        self.cot_domain_expected_benifit = {}

        # import pdb; pdb.set_trace()
        for cot_domain_name in self.cot_domain2idx:
            idx = self.cot_domain2idx[cot_domain_name]
            if idx==0:
                try:
                    cur_token = metrics[f"eval_dev_token"] # cur_token, 当前dev集上的一个平均token
                    cur_acc  = metrics[f"eval_dev_acc"] # cur_acc, 当前的dev集上的一个平均精度
                except:
                    import pdb; pdb.set_trace()
                ref_long_acc = self.target_acc[1]
                ref_long_token = self.target_token[1]
                ref_short_acc = self.target_acc[0]
                ref_short_token = self.target_token[0]
                self.cot_domain_expected_benifit[idx] = self.compute_benifit_for_shortcot(cur_token, ref_long_token, ref_short_token)
                
            if idx==1:
                try:
                    cur_token = metrics[f"eval_dev_token"] # cur_token, 当前dev集上的一个平均token
                    cur_acc  = metrics[f"eval_dev_acc"] # cur_acc, 当前的dev集上的一个平均精度
                except:
                    import pdb; pdb.set_trace()
                ref_long_acc = self.target_acc[1]
                ref_long_token = self.target_token[1]
                ref_short_acc = self.target_acc[0]
                ref_short_token = self.target_token[0]
                self.cot_domain_expected_benifit[idx] = self.compute_benifit_for_longcot(cur_acc, ref_long_acc, ref_short_acc)
            ## get current loss
            self.update_cot_domain_num += 1
        # if args.local_rank==1:
        #     import pdb; pdb.set_trace()
        update_tag = os.getenv('UPDATE_TAG', 0)
        if update_tag!='1':
            new_proportion = current_prop
        else:
            new_proportion = self.update_proportion(self.trans_dict_to_list(current_prop), self.trans_dict_to_list(self.cot_domain_expected_benifit))
        # import pdb; pdb.set_trace()
        if state.is_world_process_zero:
            wandb.log({"current_prop_id_0_short": current_prop[0]})
            wandb.log({"current_prop_id_1_long": current_prop[1]})
        self.cot_domain_expected_benifit = {}
        self.update_cot_domain_num = 0 
        # import pdb; pdb.set_trace()
        if update_tag=='1':
            new_proportion = {i: item for i, item in enumerate(new_proportion)} 
        # new_proportion = {i: item for i, item in enumerate(new_proportion)} 

        train_dataloader.dataset.update_proportion(new_proportion, state.global_step)
        train_dataloader.dataset.current_eval_dev_metrics = metrics
        eval_dataloader.dataset.start_index=0
        print(f"*** local_rank={args.local_rank}\ton evaluate finish***")
        
    def on_save(self, args, state, control, **kwargs):
        # TODO: add resume logic to ensure load correct data
        # 让save_steps承担这个角色, save_steps在这里先根据eval_dataloader的数据获得一个loss
        train_dataloader = kwargs["train_dataloader"]
        checkpoint_folder = f"checkpoint-{state.global_step}"
        # check each local_rank
        if state.is_world_process_zero:
            current_path = os.path.join(args.output_dir, checkpoint_folder)
            if not Path(current_path).exists():
                Path(current_path).mkdir(parents=True, exist_ok=True)
            with open(f"{current_path}/long2short_proportions.json", 'w') as f:
                json.dump(train_dataloader.dataset.cot_domain_proportions, f, indent=4)
            with open(f"{current_path}/eval_dev_token_acc.json", 'w') as f:
                json.dump(train_dataloader.dataset.current_eval_dev_metrics, f, indent=4) # TODO
            f.close()

        print(f"**** FOR SAVE GLOBAL_STEPS: {state.global_step}\tdomain_proportions: local_rank: {args.local_rank}\t\tdomain_proportions:{train_dataloader.dataset.cot_domain_proportions} ****")
        
    @staticmethod
    def create_instance(target_token_acc: List[float] = None,
                 set_names: List[str] = None,
                 update_type: str="doremi"
                 ):
        return partial(Long2ShortDynamicLoadingCallback, target_token_acc, set_names, update_type)

if __name__ == "__main__":
    ins = Long2ShortDynamicLoadingCallback.create_instance([(399.0, 20.0), (1233.0, 43.3)], "short_cot,long_cot", "deremi_long2short")
    print(ins().__class__)
    # print(ins.on_init_end)
    print(isinstance(ins, type))
    print(isinstance(ins(), type))











        
        

