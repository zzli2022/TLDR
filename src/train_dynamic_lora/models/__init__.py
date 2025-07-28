from typing import TYPE_CHECKING

from ..import_utils import OptionalDependencyNotAvailable, _LazyModule, is_diffusers_available


_import_structure = {
    "modeling_base": ["create_reference_model"],
    "modeling_value_head": ["AutoModelForCausalLMWithValueHead", "AutoModelForSeq2SeqLMWithValueHead"],
    "utils": [
        "prepare_deepspeed",
    ],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_sd_base"] = [
        "DDPOPipelineOutput",
        "DDPOSchedulerOutput",
        "DDPOStableDiffusionPipeline",
        "DefaultDDPOStableDiffusionPipeline",
    ]

if TYPE_CHECKING:
    from .modeling_base import GeometricMixtureWrapper, PreTrainedModelWrapper, create_reference_model
    from .modeling_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
    from .utils import (
        unwrap_model_for_generation,
    )

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_sd_base import (
            DDPOPipelineOutput,
            DDPOSchedulerOutput,
            DDPOStableDiffusionPipeline,
            DefaultDDPOStableDiffusionPipeline,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)