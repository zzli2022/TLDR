import inspect
import os
import warnings
from contextlib import contextmanager
import sys


import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin

from .tuners import LoraModel, PrefixEncoder, PromptEmbedding, PromptEncoder
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
    _set_trainable,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    shift_tokens_right,
)


class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Parameter-Efficient Fine-Tuning Model. Base model encompassing various Peft methods.

    Args:
        model ([`PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.


    **Attributes**:
        - **base_model** ([`PreTrainedModel`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
        `isinstance(self.peft_config, PromptLearningConfig)`.
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
        `isinstance(self.peft_config, PromptLearningConfig)`.
        - **transformer_backbone_name** (`str`) -- The name of the transformer
        backbone in the base model if `isinstance(self.peft_config, PromptLearningConfig)`.
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
        in the base model if `isinstance(self.peft_config, PromptLearningConfig)`.
    """

    def __init__(self, model, peft_config: PeftConfig): # casualLM, LoraConfig
        super().__init__()
        self.peft_config = peft_config
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        if isinstance(self.peft_config, PromptLearningConfig):
            self._setup_prompt_encoder()
        else: # --------------> here
            self.base_model = LoraModel(peft_config, model)
        if getattr(self.peft_config, "modules_to_save", None) is not None:
            self.modules_to_save = self.peft_config.modules_to_save
            _set_trainable(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        Args:
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        re-loaded using the `LoraModel.from_pretrained` class method, and also used by the `LoraModel.push_to_hub`
        method.
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            **kwargs:
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if self.peft_config.base_model_name_or_path is None:
            self.peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if isinstance(self.peft_config, PromptLearningConfig)
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = self.peft_config.inference_mode
        self.peft_config.inference_mode = True
        self.peft_config.save_pretrained(save_directory)
        self.peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        r"""
        Args:
        Instantiate a `LoraModel` from a pretrained Lora configuration and weights.
            model (`transformers.PreTrainedModel`):
                The model to be adapted. The model should be initialized with the `from_pretrained` method. from
                `transformers` library.
            model_id (`str`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on
                        huggingface Hub
                    - A path to a directory containing a Lora configuration file saved using the
                        `save_pretrained` method, e.g., ``./my_lora_config_directory/``.
        """
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig.from_pretrained(model_id).peft_type].from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config)
        else: # -----------------> here
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)
            
        if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
            filename = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME)
            except:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)


        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model

    def _setup_prompt_encoder(self):
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if self.peft_config.num_transformer_submodules is None:
            self.peft_config.num_transformer_submodules = (
                2 if self.peft_config.task_type == TaskType.SEQ_2_SEQ_LM else 1
            )

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if self.peft_config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(self.peft_config, self.word_embeddings)
        elif self.peft_config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(self.peft_config)
        elif self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(self.peft_config)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder = prompt_encoder
        self.prompt_tokens = torch.arange(
            self.peft_config.num_virtual_tokens * self.peft_config.num_transformer_submodules
        ).long()

    def get_prompt_embedding_to_save(self):
        """
        Returns the prompt embedding to save when saving the model. Only applicable when `peft_config.peft_type !=
        PeftType.LORA`.
        """
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(1, -1).to(self.device)
        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config.num_virtual_tokens]
        prompt_embeddings = self.prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size):
        """
        Returns the virtual prompts to use for Peft. Only applicable when `peft_config.peft_type != PeftType.LORA`.
        """
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config.num_virtual_tokens]
            if self.peft_config.inference_mode:
                past_key_values = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = self.prompt_encoder(prompt_tokens)
            past_key_values = past_key_values.view(
                batch_size,
                self.peft_config.num_virtual_tokens,
                self.peft_config.num_layers * 2,
                self.peft_config.num_attention_heads,
                self.peft_config.token_dim // self.peft_config.num_attention_heads,
            )
            if self.peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                self.peft_config.num_transformer_submodules * 2
            )
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if self.peft_config.inference_mode:
                prompts = self.prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = self.prompt_encoder(prompt_tokens)
            return prompts

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                # print(f'trainable, name: {_}, params: {param}')
                trainable_params += num_params
            # else:
            #     print(f'freeze, name: {_}, params: {param}')
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs):  # pylint: disable=E0202
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        if isinstance(self.peft_config, PromptLearningConfig):
            old_forward = self.forward
            self.forward = self.base_model.forward
        else:
            self.base_model.disable_adapter_layers()
        yield
        if isinstance(self.peft_config, PromptLearningConfig):
            self.forward = old_forward
        else:
            self.base_model.enable_adapter_layers()

    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model if isinstance(self.peft_config, PromptLearningConfig) else self.base_model.model

class PeftModelForSequenceClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example::

        >>> from transformers import AutoModelForSequenceClassification >>> from peft import
        PeftModelForSequenceClassification, get_peft_config >>> config = {
                'peft_type': 'PREFIX_TUNING', 'task_type': 'SEQ_CLS', 'inference_mode': False, 'num_virtual_tokens':
                20, 'token_dim': 768, 'num_transformer_submodules': 1, 'num_attention_heads': 12, 'num_layers': 12,
                'encoder_hidden_size': 768, 'prefix_projection': False, 'postprocess_past_key_value_function': None
            }
        >>> peft_config = get_peft_config(config) >>> model =
        AutoModelForSequenceClassification.from_pretrained("bert-base-cased") >>> peft_model =
        PeftModelForSequenceClassification(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self)

    def forward(    # pylint: disable=W0221
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, self.peft_config.num_virtual_tokens).to(self.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

class PeftModelForCausalLM(PeftModel):
    """
    Peft model for Causal LM

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.


    Example::

        >>> from transformers import AutoModelForCausalLM >>> from peft import PeftModelForCausalLM, get_peft_config
        >>> config = {
                'peft_type': 'PREFIX_TUNING', 'task_type': 'CAUSAL_LM', 'inference_mode': False, 'num_virtual_tokens':
                20, 'token_dim': 1280, 'num_transformer_submodules': 1, 'num_attention_heads': 20, 'num_layers': 36,
                'encoder_hidden_size': 1280, 'prefix_projection': False, 'postprocess_past_key_value_function': None
            }
        >>> peft_config = get_peft_config(config) >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large") >>>
        peft_model = PeftModelForCausalLM(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
    """

    def __init__(self, model, peft_config: PeftConfig): # casualLM, LoraConfig
        super().__init__(model, peft_config)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(# pylint: disable=W0221
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_types=None,
        **kwargs,
    ):
        if not isinstance(self.peft_config, PromptLearningConfig): # here
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task_types=task_types,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.peft_config.num_virtual_tokens), -100).to(self.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(self.peft_config, PromptLearningConfig):
            if model_kwargs["past_key_values"] is None and self.peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs

class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for Seq2Seq LM

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.


    Example::

        >>> from transformers import AutoModelForSeq2SeqLM >>> from peft import PeftModelForSeq2SeqLM, get_peft_config
        >>> config = {
                'peft_type': 'LORA', 'task_type': 'SEQ_2_SEQ_LM', 'inference_mode': False, 'r': 8, 'target_modules':
                ['q', 'v'], 'lora_alpha': 32, 'lora_dropout': 0.1, 'merge_weights': False, 'fan_in_fan_out': False,
                'enable_lora': None, 'bias': 'none'
            }
        >>> peft_config = get_peft_config(config) >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>>
        peft_model = PeftModelForSeq2SeqLM(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(self.device)
            decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(self.device)
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                if self.peft_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif self.peft_config.num_transformer_submodules == 2:
                    prefix_labels = torch.full((batch_size, self.peft_config.num_virtual_tokens), -100).to(self.device)
                    kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : self.peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            if self.peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif self.peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = torch.cat(
                    (prompts[:, self.peft_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
                )
                return self.base_model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
                )

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if not isinstance(self.peft_config, PromptLearningConfig):
                outputs = self.base_model.generate(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
                    outputs = self.base_model.generate(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if model_kwargs["past_key_values"] is None and self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs["past_key_values"] = past_key_values
        return model_kwargs

class PeftModelForTokenClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example::

        >>> from transformers import AutoModelForSequenceClassification >>> from peft import
        PeftModelForTokenClassification, get_peft_config >>> config = {
                'peft_type': 'PREFIX_TUNING', 'task_type': 'TOKEN_CLS', 'inference_mode': False, 'num_virtual_tokens':
                20, 'token_dim': 768, 'num_transformer_submodules': 1, 'num_attention_heads': 12, 'num_layers': 12,
                'encoder_hidden_size': 768, 'prefix_projection': False, 'postprocess_past_key_value_function': None
            }
        >>> peft_config = get_peft_config(config) >>> model =
        AutoModelForTokenClassification.from_pretrained("bert-base-cased") >>> peft_model =
        PeftModelForTokenClassification(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, self.peft_config.num_virtual_tokens).to(self.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)

            loss = None
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
