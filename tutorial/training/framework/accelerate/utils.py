import os

from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from torch import nn


def save_checkpoint(model: nn.Module, accelerator: Accelerator, output_dir: str):
    # for models from Hugging Face, we need to unwrap the model to use its "save_pretrained" method so that it can be loaded by using "from_pretrained"
    unwrapped_model = accelerator.unwrap_model(model)

    # get the state_dict of the model by using the accelerator because it handles the device placement automatically
    state_dict = accelerator.get_state_dict(model)
    if hasattr(unwrapped_model, "save_pretrained"):
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            state_dict=state_dict,
            save_function=accelerator.save,
        )
    else:
        save_file(state_dict, os.path.join(output_dir, "pytorch_model.safetensors"))


def load_checkpoint(model: nn.Module, input_dir: str):
    if hasattr(model, "from_pretrained"):
        model.from_pretrained(input_dir)
    else:
        state_dict = load_file(os.path.join(input_dir, "pytorch_model.safetensors"))
        model.load_state_dict(state_dict)
