from transformers import Trainer as HFTrainer
from transformers.trainer import logger
from transformers.trainer_callback import (
    PrinterCallback,
    TrainerCallback,
)
import torch
import torch.distributed as dist
import time


class LogCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.last_log_time = None
        self.log_time_interval = 0
        self.is_training = False

        self.max_steps = -1
        self.first_step_of_run = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.last_log_time is None:
            self.log_time_interval = getattr(args, "log_time_interval", 0)
            if self.log_time_interval > 0:
                logger.info(f"Using log_time_interval {self.log_time_interval} s. This will override logging_steps and logging_strategy.")
                args.logging_steps = 1
                args.logging_strategy = "steps"

            self.last_step = 0

            self.start_time = time.time()
            self.last_log_time = self.start_time
            self.max_steps = state.max_steps
            self.first_step_of_run = state.global_step

            self.last_tokens_seen = state.num_input_tokens_seen

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)

        if state.is_world_process_zero:
            if self.is_training:
                current_time = time.time()
                time_diff = current_time - self.last_log_time
                force = logs.get("force", False)

                if time_diff > self.log_time_interval or state.global_step >= self.max_steps - 1 or force:
                    self.last_log_time = current_time
                    steps_completed = max(state.global_step, 1)

                    steps_since_first = max(1, state.global_step - self.first_step_of_run)
                    self.last_step = state.global_step

                    tokens_seen_since_last = (state.num_input_tokens_seen - self.last_tokens_seen) // args.seq_parallel_size
                    self.last_tokens_seen = state.num_input_tokens_seen

                    remaining_steps = self.max_steps - steps_completed
                    pct_completed = (steps_completed / self.max_steps) * 100
                    time_since_start = current_time - self.start_time
                    remaining_time = (time_since_start / steps_since_first) * remaining_steps

                    gpu_mem_free, _ = torch.cuda.mem_get_info(device=args.device)

                    update = {
                        "completed": f"{pct_completed:.2f}% ({steps_completed:_} / {self.max_steps:_})",
                        "remaining time": self.format_duration(remaining_time),
                        "throughput": f"{tokens_seen_since_last / time_diff:.2f}",
                        "gpu_mem_free": f"{gpu_mem_free / 1024 / 1024:.0f}MB",
                    }

                    logger.info(str({**logs, **update}))
            else:
                logger.info(str(logs))

    def on_train_begin(self, args, state, control, **kwargs):
        args.include_num_input_tokens_seen = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.is_training = True

    def on_prediction_step(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.is_training = False

    @staticmethod
    def format_duration(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"


class Trainer(HFTrainer):
    def __init__(self, model, args, *more_args, **kwargs):
        super().__init__(model, args, *more_args, **kwargs)
        # import ipdb; ipdb.set_trace()
        if not dist.is_initialized() or args.seq_parallel_size == dist.get_world_size():
            logger.info(f"Using world as sequence parallel group")
            self.seq_parallel_group = dist.group.WORLD
        else:
            logger.info(f"Initializing sequence parallel groups with size {args.seq_parallel_size}")
            self.seq_parallel_group, _ = dist.new_subgroups(args.seq_parallel_size)

        try:
            self.remove_callback(PrinterCallback)
            self.add_callback(LogCallback)
            # self.add_callback(SIGUSR1Callback(self))
        except ValueError:
            logger.warn("Couldn't remove PrinterCallback")

    def get_sequence_parallel_inputs(self, inputs):
        seq_parallel_world_size = (dist.get_world_size(self.seq_parallel_group) if dist.is_initialized() else 1)
        num_items_in_batch = inputs["num_items_in_batch"]
        if seq_parallel_world_size > 1:
            seq_parallel_rank = dist.get_rank(self.seq_parallel_group)

            input_ids = inputs["input_ids"]
            labels = inputs["labels"]

            shifted_labels = labels
            # shifted_labels = labels.roll(-1, dims=-1)
            # shifted_labels[..., -1] = -100
            seq_lengths = seq_lengths=torch.tensor([input_ids.size(1)], dtype=torch.long)

            # add right padding here to make equal sized chunks
            if input_ids.size(-1) % seq_parallel_world_size != 0:
                padding = seq_parallel_world_size - (input_ids.size(-1) % seq_parallel_world_size)
                padding_zeros = torch.full(input_ids.size()[:-1] + (padding,), 0, dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, padding_zeros], dim=-1)
                shifted_labels = torch.cat([shifted_labels, padding_zeros-100], dim=-1)
                seq_lengths[-1] += padding

            # select chunk of input_ids and labels
            input_ids_chunks = torch.tensor_split(input_ids, seq_parallel_world_size, dim=-1)
            shifted_labels_chunks = torch.tensor_split(shifted_labels, seq_parallel_world_size, dim=-1)

            inputs = {
                "input_ids": input_ids_chunks[seq_parallel_rank],
                "labels": shifted_labels_chunks[seq_parallel_rank],
                "seq_lengths": seq_lengths,
                "seq_parallel_group": self.seq_parallel_group,
                "num_items_in_batch": num_items_in_batch,
            }

            max_seq_length = seq_lengths.max()
            max_tokens_per_device = seq_lengths.sum() // seq_parallel_world_size

            start_index = sum(chunk.size(-1) for chunk in input_ids_chunks[:seq_parallel_rank])
            end_index = start_index + input_ids_chunks[seq_parallel_rank].size(-1)
            # inputs["position_ids"] = torch.tensor([start_index]).to(input_ids.device)
            inputs["position_ids"] = torch.range(start_index, end_index - 1, dtype=torch.long).to(input_ids.device).unsqueeze(0)
            # max sequence length is smaller per device => no need for sequence parallelism
            if max_seq_length <= max_tokens_per_device:
                # take the seq length field and only retain seq lengths with indices that are valid for this rank
                seq_indices = seq_lengths.cumsum(-1)
                seq_indices = seq_indices[(seq_indices < end_index) & (seq_indices >= start_index)]

                start_index_tensor = torch.tensor([start_index], device=seq_indices.device)
                end_index_tensor = torch.tensor([end_index], device=seq_indices.device)

                seq_lengths = seq_indices.diff(prepend=start_index_tensor, append=end_index_tensor)
                seq_lengths = seq_lengths[seq_lengths > 0]
                inputs["seq_lengths"] = seq_lengths
                inputs["seq_parallel_group"] = None

        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
    
        inputs = self.get_sequence_parallel_inputs(inputs)
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss