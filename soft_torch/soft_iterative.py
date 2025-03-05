import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF

from transformers import AutoModelForCausalLM, AutoConfig

from loguru import logger


@dataclass
class SOFTConfig:
    r: int
    lora_only: bool = False
    trainable_scaling: bool = False
    quantize: str = None
    dtype: str = "float32"
    use_double_quant: bool = False


class SOFTModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        r=8,
        keep_original_weights=True,
        trainable_scaling=False,
        quantize=None,
        dtype="float32",
        use_double_quant=False,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        super().__init__()

        for param in model.parameters():
            param.requires_grad = False

        self.wrapped_model: nn.Module = model
        self.r = r
        self.trainable_scaling = trainable_scaling

        self._config = SOFTConfig(
            r=r,
            quantize=quantize,
            use_double_quant=use_double_quant,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            weight_data = module.weight.data
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data

            new_module = SOFTLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                r=self.r,
                quantize=quantize,
                dtype=dtype,
                weight_data=weight_data,
                bias_data=bias_data,
            )

            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent


    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "relora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "relora_config.json"), "r") as f:
            relora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in relora_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {relora_config['keep_original']}")
            relora_config["lora_only"] = not relora_config.pop("keep_original")
            relora_config["keep_original_weights"] = not relora_config["lora_only"]

        if "trainable_scaling" not in relora_config:
            relora_config["trainable_scaling"] = False

        model = cls(base_model, **relora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class SOFTLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        weight_data=None,
        bias_data=None,
        # trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
        quantize=False,
        # bnb_4bit_use_double_quant=False,
        # bnb_4bit_quant_type="nf4",
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """
        nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        # if full model weight + lora weight
        if bias_data is None:
            bias_data = torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True) if bias else None
        self.bias = nn.Parameter(bias_data) if bias else None

        if weight_data is None:
            # note that our trainable weight are W_a and W_b
            weight_data = torch.zeros(out_features, in_features, device=device, dtype=dtype, requires_grad=False)

        if quantize is None:
            self.weight = nn.Parameter(weight_data, requires_grad=False)
        elif quantize == "4bit":
            self.weight = bnb.nn.Params4bit(
                weight_data,
                requires_grad=False,
                compress_statistics=bnb_4bit_use_double_quant,
                quant_type=bnb_4bit_quant_type,
            )
        elif quantize == "8bit":
            logger.warning("Int8 currently does not support merge_and_reinit! It will fail")
            self.weight = bnb.nn.Int8Params(
                weight_data,
                requires_grad=False,
            )
        else:
            raise ValueError(f"Unknown quantize type: {quantize}")


        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        # self.lora_dropout = nn.Dropout(p=lora_dropout)
        # self.trainable_scaling = trainable_scaling
        self.quantize = quantize
        self.dtype = dtype

        if dtype in ["bf16", "bfloat16"]:
            self.soft_R = nn.Parameter(torch.eye(self.r, dtype=torch.bfloat16))
        else:
            self.soft_R = nn.Parameter(torch.eye(self.r))
        self.register_buffer("soft_R_indices", torch.arange(self.r))
        #if trainable_scaling:
        #    self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        #else:
        #    self.scaling = self.lora_alpha / self.r

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

    @torch.no_grad()
    def merge_and_reinit(self):
        new_weight = self.weight.data
        new_weight = torch.transpose(new_weight, 0, 1)
        rotated_weight = torch.mm(self.soft_R, new_weight[self.soft_R_indices, :])
        new_weight[self.soft_R_indices, :] = rotated_weight
        new_weight = torch.transpose(new_weight, 0, 1)

        self.weight.data = new_weight
        self.soft_R.copy_(torch.eye(self.r, dtype=self.soft_R.dtype))

    @torch.no_grad()
    def sample_dimensions(self):
        self.soft_R_indices = torch.randperm(self.in_features)[:self.r]

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            self.merge_and_reinit()

        self.sample_dimensions()

        if self.quantize == "4bit":
            result = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        elif self.quantize == "8bit":
            result = bnb.matmul(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        else:
            new_weight = self.weight.data
            new_weight = torch.transpose(new_weight, 0, 1)
            rotated_weight = torch.mm(self.soft_R, new_weight[self.soft_R_indices, :])
            new_weight[self.soft_R_indices, :] = rotated_weight
            new_weight = torch.transpose(new_weight, 0, 1)

            # Apply the trainable identity matrix
            result = F.linear(input=x, weight=new_weight, bias=self.bias)

        return result