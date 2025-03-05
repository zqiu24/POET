import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF

from transformers import AutoModelForCausalLM, AutoConfig

from loguru import logger


@dataclass
class SOFTConfig:
    r: int
    target_modules: List[str]
    soft_only: bool = False
    trainable_scaling: bool = False
    quantize: str = None
    dtype: str = "float32"
    use_double_quant: bool = False


class SOFTModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=8,
        optim_loop=20,
        trainable_scaling=False,
        quantize=None,
        dtype="float32",
        use_double_quant=False,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        super().__init__()

        self.wrapped_model: nn.Module = model
        self.r = r
        self.trainable_scaling = trainable_scaling
        self.quantize = quantize
        self.dtype = dtype

        self._config = SOFTConfig(
            r=r,
            target_modules=target_modules,
            quantize=quantize,
            use_double_quant=use_double_quant,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        for module_name, module in self.wrapped_model.named_modules():
            # if not any(target_key in module_name for target_key in target_modules_list):
            #     print(f"Skipping {module_name}")
            #     continue

            if isinstance(module, nn.Linear):
                weight_data = module.weight.data
                bias_data = None
                if module.bias is not None:
                    bias_data = module.bias.data

                new_module = SOFTLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    r=r,
                    optim_loop=optim_loop,
                    quantize=quantize,
                    dtype=dtype,
                    weight_data=weight_data,
                    bias_data=bias_data,
                )
            else:
                continue
            '''
            elif isinstance(module, nn.Embedding):
                weight_data = module.weight.data

                new_module = SOFTEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    r=r,
                    optim_loop=optim_loop,
                    quantize=quantize,
                    dtype=dtype,
                    weight_data=weight_data,
                )
            '''

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


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class SOFTLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        optim_loop: int,
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

        if quantize is None:
            self.weight = nn.Parameter(weight_data, requires_grad=False)
            init.kaiming_normal_(self.weight)
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
        # self.r = r
        self.r = in_features
        self.optim_loop = optim_loop
        # self.lora_dropout = nn.Dropout(p=lora_dropout)
        # self.trainable_scaling = trainable_scaling
        self.quantize = quantize
        self.dtype = dtype
        self.iteration_count = 0

        if dtype in ["bf16", "bfloat16"]:
            self.soft_R = nn.Parameter(torch.zeros(self.r, self.r, dtype=torch.bfloat16))
            # self.register_buffer("soft_R_weight", torch.eye(self.r, dtype=torch.bfloat16))
        else:
            self.soft_R = nn.Parameter(torch.zeros(self.r, self.r))
            # self.register_buffer("soft_R_weight", torch.eye(self.r))
        self.register_buffer("soft_R_indices", torch.arange(self.r))
        #if trainable_scaling:
        #    self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        #else:
        #    self.scaling = self.lora_alpha / self.r

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

    @torch.no_grad()
    def merge_and_reinit(self):
        soft_R_weight = self.cayley(self.soft_R)
        new_weight = self.weight.data
        new_weight = torch.transpose(new_weight, 0, 1)
        rotated_weight = torch.mm(soft_R_weight, new_weight[self.soft_R_indices, :])
        new_weight[self.soft_R_indices, :] = rotated_weight
        new_weight = torch.transpose(new_weight, 0, 1)

        # print('is orthogonal:', self.is_orthogonal(soft_R_weight))
        # print('is identity:', self.is_identity_matrix(soft_R_weight))

        self.weight.data = new_weight
        nn.init.zeros_(self.soft_R)

    @torch.no_grad()
    def sample_dimensions(self):
        self.soft_R_indices = torch.randperm(self.in_features)[:self.r].to(self.soft_R_indices.device)

    def cayley(self, w):
        if self.dtype in ["bf16", "bfloat16"]:
            w = w.to(torch.float32)
            # Ensure the input matrix is skew-symmetric
            skew = 0.5 * (w - w.t())
            I = torch.eye(w.shape[0], device=w.device)
            # Perform the Cayley parametrization
            Q = torch.mm(I + skew, torch.inverse(I - skew))
            Q = Q.to(torch.bfloat16)
        else:
            # Ensure the input matrix is skew-symmetric
            skew = 0.5 * (w - w.t())
            I = torch.eye(w.shape[0], device=w.device)
            # Perform the Cayley parametrization
            Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q
    
    def iterative_cayley(self, w, n_iter=5):
        for _ in range(n_iter):
            w = self.cayley(w)
        return w

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)
        
    def is_identity_matrix(self, R):
        if not torch.is_tensor(R):
            raise TypeError("Input must be a PyTorch tensor.")
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            return False
        identity = torch.eye(R.shape[0], device=R.device)
        return torch.all(torch.eq(R, identity))

    def forward(self, x: torch.Tensor):
        self.iteration_count += 1

        '''
        if self.iteration_count % self.optim_loop == 0:
            self.merge_and_reinit()
            self.sample_dimensions()
        '''
        soft_R_weight = self.cayley(self.soft_R)
        # print('is orthogonal:', self.is_orthogonal(self.soft_R))
        # print('is identity:', self.is_identity_matrix(self.soft_R))

        # print('soft R weight dtype:', soft_R_weight.dtype)
        # print('self.soft R dtype:', self.soft_R.dtype)

        if self.quantize == "4bit":
            result = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        elif self.quantize == "8bit":
            result = bnb.matmul(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        else:
            new_weight = self.weight.transpose(0, 1)
            rotated_weight = torch.mm(soft_R_weight, new_weight)
            updated_weight = torch.transpose(rotated_weight, 0, 1)

            result = F.linear(input=x, weight=updated_weight, bias=self.bias)
            '''
            new_weight = self.weight.transpose(0, 1)
            rotated_weight = torch.mm(soft_R_weight, new_weight[self.soft_R_indices, :])
            expanded_indices = self.soft_R_indices.unsqueeze(1).expand(-1, rotated_weight.shape[1])
            updated_weight = new_weight.scatter(0, expanded_indices, rotated_weight)
            updated_weight = torch.transpose(updated_weight, 0, 1)

            result = F.linear(input=x, weight=updated_weight, bias=self.bias)
            '''
        return result
    

class SOFTEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int,
        optim_loop: int,
        *,
        weight_data=None,
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

        if quantize is None:
            self.weight = nn.Parameter(weight_data, requires_grad=False)
            init.kaiming_normal_(self.weight)
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

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # self.r = r
        self.r = embedding_dim
        self.optim_loop = optim_loop
        # self.lora_dropout = nn.Dropout(p=lora_dropout)
        # self.trainable_scaling = trainable_scaling
        self.quantize = quantize
        self.dtype = dtype
        self.iteration_count = 0

        if dtype in ["bf16", "bfloat16"]:
            self.soft_R = nn.Parameter(torch.zeros(self.r, self.r, dtype=torch.bfloat16))
            # self.register_buffer("soft_R_weight", torch.eye(self.r, dtype=torch.bfloat16))
        else:
            self.soft_R = nn.Parameter(torch.zeros(self.r, self.r))
            # self.register_buffer("soft_R_weight", torch.eye(self.r))
        self.register_buffer("soft_R_indices", torch.arange(self.r))
        #if trainable_scaling:
        #    self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        #else:
        #    self.scaling = self.lora_alpha / self.r

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

    @torch.no_grad()
    def merge_and_reinit(self):
        soft_R_weight = self.cayley(self.soft_R)
        new_weight = self.weight.data
        new_weight = torch.transpose(new_weight, 0, 1)
        rotated_weight = torch.mm(soft_R_weight, new_weight[self.soft_R_indices, :])
        new_weight[self.soft_R_indices, :] = rotated_weight
        new_weight = torch.transpose(new_weight, 0, 1)

        # print('is orthogonal:', self.is_orthogonal(soft_R_weight))
        # print('is identity:', self.is_identity_matrix(soft_R_weight))

        self.weight.data = new_weight
        nn.init.zeros_(self.soft_R)

    @torch.no_grad()
    def sample_dimensions(self):
        self.soft_R_indices = torch.randperm(self.embedding_dim)[:self.r].to(self.soft_R_indices.device)

    def cayley(self, w):
        if self.dtype in ["bf16", "bfloat16"]:
            w = w.to(torch.float32)
            # Ensure the input matrix is skew-symmetric
            skew = 0.5 * (w - w.t())
            I = torch.eye(w.shape[0], device=w.device)
            # Perform the Cayley parametrization
            Q = torch.mm(I + skew, torch.inverse(I - skew))
            Q = Q.to(torch.bfloat16)
        else:
            # Ensure the input matrix is skew-symmetric
            skew = 0.5 * (w - w.t())
            I = torch.eye(w.shape[0], device=w.device)
            # Perform the Cayley parametrization
            Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q
    
    def iterative_cayley(self, w, n_iter=5):
        for _ in range(n_iter):
            w = self.cayley(w)
        return w

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)
        
    def is_identity_matrix(self, R):
        if not torch.is_tensor(R):
            raise TypeError("Input must be a PyTorch tensor.")
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            return False
        identity = torch.eye(R.shape[0], device=R.device)
        return torch.all(torch.eq(R, identity))

    def forward(self, x: torch.Tensor):
        self.iteration_count += 1
        
        if self.iteration_count % self.optim_loop == 0:
            self.merge_and_reinit()
            self.sample_dimensions()

        soft_R_weight = self.cayley(self.soft_R)
        # print('is orthogonal:', self.is_orthogonal(self.soft_R))
        # print('is identity:', self.is_identity_matrix(self.soft_R))

        # print('soft R weight dtype:', soft_R_weight.dtype)
        # print('self.soft R dtype:', self.soft_R.dtype)

        if self.quantize == "4bit":
            result = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        elif self.quantize == "8bit":
            result = bnb.matmul(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        else:
            new_weight = self.weight.transpose(0, 1)
            rotated_weight = torch.mm(soft_R_weight, new_weight[self.soft_R_indices, :])
            expanded_indices = self.soft_R_indices.unsqueeze(1).expand(-1, rotated_weight.shape[1])
            updated_weight = new_weight.scatter(0, expanded_indices, rotated_weight)
            updated_weight = torch.transpose(updated_weight, 0, 1)
            result = F.embedding(input=x, weight=updated_weight)

        return result