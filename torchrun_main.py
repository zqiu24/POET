import os
import time
import json
import random
import argparse
import numpy as np
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
from torch.optim import Optimizer

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
from datasets import DownloadConfig, Features, Value
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils, memory_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
from soft_torch import matmul, SkewSymmetric

transformers.logging.set_verbosity_error()

import warnings
from typing import Callable, Iterable, Optional, Tuple, Union
class SOFTAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias, 
                   "stochastic": False, "update_reset_R_gap": 0}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # Check if reset should be applied based on CayleyLinear's global counter
                if group.get("stochastic", False) and group.get("update_reset_R_gap", 0) > 0 and \
                   CayleyLinear.global_step_counter % group["update_reset_R_gap"] == 0 and \
                   CayleyLinear.global_step_counter > 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


def find_closest_valid_rank(in_features, out_features, target_rank):
    """Find the closest rank that divides both in_features and out_features evenly."""
    # Get all factors of both in_features and out_features
    def get_factors(n):
        factors = []
        for i in range(1, int(n ** 0.5) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        return sorted(factors)
    
    in_factors = get_factors(in_features)
    out_factors = get_factors(out_features)
    
    # Find common factors
    common_factors = sorted(list(set(in_factors) & set(out_factors)))
    
    # Find closest factor to target_rank
    closest_rank = min(common_factors, key=lambda x: abs(x - target_rank))
    return closest_rank


class CayleyLinear(nn.Linear):
    # Class-level variable to track steps across all instances
    global_step_counter = 0
    
    def __init__(self, linear_layer, rank=128, stochastic=False):
        """Initialize CayleyLinear by inheriting from an existing linear layer."""
        super().__init__(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
        )
        
        # Move the layer to the correct device first
        self.to(linear_layer.weight.device)
        
        # Remove the inherited weight parameter
        del self.weight
        
        # Register the weight as a buffer instead
        self.register_buffer('weight', linear_layer.weight.data.clone())
        
        # Handle bias normally since it remains trainable
        if self.bias is not None:
            self.bias.data.copy_(linear_layer.bias.data)
        
        self.use_neumann = False
        self.num_neumann_terms = 1
        self.reset_R = False
        self.update_reset_R_gap = 0
        self.stochastic = stochastic

        self.rank = rank  # Store original requested rank
        if self.out_features % rank != 0 or self.in_features % rank != 0:
            original_rank = rank
            self.rank = find_closest_valid_rank(self.in_features, self.out_features, rank)
            warnings.warn(
                f"Requested rank {original_rank} is not valid as it doesn't divide both "
                f"input features ({self.in_features}) and output features ({self.out_features}) evenly. "
                f"Using closest valid rank: {self.rank} instead."
            )
            rank = self.rank  # Update rank for subsequent calculations
            print(f"self.in_features: {self.in_features}")
            print(f"self.out_features: {self.out_features}")
            print(f"Using rank: {self.rank} instead of {original_rank}")
            exit()
        
        if self.stochastic:
            self.block_num = 1
        else:
            self.block_num = self.rank
        self.d_out = self.out_features // self.rank
        self.d_in = self.in_features // self.rank
        n_elements_out = (self.d_out * (self.d_out - 1)) // 2
        n_elements_in = (self.d_in * (self.d_in - 1)) // 2

        # op = matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device=device)
        if self.rank > 10000:
            layout = torch.eye(self.rank).unsqueeze(0).to(torch.int64)
            self.op_left = matmul(layout, self.d_out, 'qoft_dsd', trans_a=False, trans_b=False, device=linear_layer.weight.device)
            self.op_right = matmul(layout, self.d_in, 'qoft_dds', trans_a=False, trans_b=False, device=linear_layer.weight.device)

        # Pre-compute and store triu_indices as buffers
        i_out, j_out = torch.triu_indices(self.d_out, self.d_out, offset=1)
        self.triu_indices_out_i = i_out
        self.triu_indices_out_j = j_out
        
        i_in, j_in = torch.triu_indices(self.d_in, self.d_in, offset=1)
        self.triu_indices_in_i = i_in
        self.triu_indices_in_j = j_in
        
        if self.stochastic:
            self.update_indices()

            self.Q_left = nn.Parameter(torch.zeros(n_elements_out,
                                device=linear_layer.weight.device,
                                dtype=linear_layer.weight.dtype))
            self.Q_right = nn.Parameter(torch.zeros(n_elements_in,
                            device=linear_layer.weight.device,
                            dtype=linear_layer.weight.dtype))
            
            '''
            self.Q_left = nn.Parameter(torch.zeros(1, self.d_out, self.d_out,
                                device=linear_layer.weight.device,
                                dtype=linear_layer.weight.dtype))
            self.Q_right = nn.Parameter(torch.zeros(1, self.d_in, self.d_in,
                            device=linear_layer.weight.device,
                            dtype=linear_layer.weight.dtype))
            '''
        else:
            # Store only the upper triangular elements in a 1D tensor
            self.Q_left = nn.Parameter(torch.zeros(self.rank, n_elements_out,
                                device=linear_layer.weight.device,
                                dtype=linear_layer.weight.dtype))
            # Store only the upper triangular elements in a 1D tensor
            self.Q_right = nn.Parameter(torch.zeros(self.rank, n_elements_in,
                            device=linear_layer.weight.device,
                            dtype=linear_layer.weight.dtype))          

        # print(f"self.out_features: {self.out_features}")
        # print(f"self.in_features: {self.in_features}")

        # self.scale_init = 1.0  # Actual initial value you want
        # self.scale_scale = 1.0 / math.sqrt(self.out_features)  # Controls effective learning rate
        # self.scale = nn.Parameter(torch.ones(self.out_features) * self.scale_scale, device=linear_layer.weight.device, dtype=linear_layer.weight.dtype)
        # Initialize scaling factors - per neuron scaling
        # self.scale_init = 1.0
        # self.scale_scale = 1.0 / math.sqrt(self.out_features)
        # Initialize scale as a parameter with shape [out_features, 1]
        # self.scale = nn.Parameter(torch.randn(self.out_features, 1, 
        #                                    device=linear_layer.weight.device, 
        #                                    dtype=linear_layer.weight.dtype) * 0.01)
        # self.s = nn.Parameter(torch.ones(self.out_features, device=linear_layer.weight.device, dtype=linear_layer.weight.dtype))

    
    def update_indices(self):
        """Update the selected indices randomly."""
        with torch.no_grad():
            self.selected_indices_left = torch.randperm(self.out_features, device=self.weight.device)[:self.d_out]
            self.selected_indices_right = torch.randperm(self.in_features, device=self.weight.device)[:self.d_in]
    
    
    def get_cayley_transform(self, mode='all'):
        """
        Get Cayley transform matrices based on specified mode.
        Args:
            mode (str): One of 'all', 'left', or 'right' to specify which transforms to compute
        Returns:
            Tuple of (R_left, R_right) where each R is a batch of smaller orthogonal matrices
        """
        R_left = None
        R_right = None
        
        # Process left transform if needed
        if mode in ['all', 'left']:
            Q_left = self.Q_left.to(torch.float32)
            
            # Create skew-symmetric matrix using block_num and d_out
            Q_blocks = torch.zeros(self.block_num, self.d_out, self.d_out, device=self.Q_left.device, dtype=torch.float32)
            Q_blocks[:, self.triu_indices_out_i, self.triu_indices_out_j] = Q_left
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Cayley transform
            I = torch.eye(self.d_out, device=self.Q_left.device, dtype=torch.float32)
            R_left = torch.linalg.solve(I - Q_blocks, I + Q_blocks).transpose(-2, -1)
            R_left = R_left.to(self.Q_left.dtype)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            Q_right = self.Q_right.to(torch.float32)
            
            # Create skew-symmetric matrix using block_num and d_in
            Q_blocks = torch.zeros(self.block_num, self.d_in, self.d_in, device=self.Q_right.device, dtype=torch.float32)
            Q_blocks[:, self.triu_indices_in_i, self.triu_indices_in_j] = Q_right
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Cayley transform
            I = torch.eye(self.d_in, device=self.Q_right.device, dtype=torch.float32)
            R_right = torch.linalg.solve(I - Q_blocks, I + Q_blocks).transpose(-2, -1)
            R_right = R_right.to(self.Q_right.dtype)

        return R_left, R_right

    def get_cayley_transform_neumann(self, mode='all'):
        """
        Get Cayley transform matrices using Neumann series approximation.
        Args:
            mode (str): One of 'all', 'left', or 'right' to specify which transforms to compute
        Returns:
            Tuple of (R_left, R_right) where each R is a batch of smaller orthogonal matrices
        """
        R_left = None
        R_right = None
        
        # Process left transform if needed
        if mode in ['all', 'left']:
            Q_left = self.Q_left.to(torch.float32)
            
            # Create skew-symmetric matrix using block_num and d_out
            Q_blocks = torch.zeros(self.block_num, self.d_out, self.d_out, device=self.Q_left.device, dtype=torch.float32)
            Q_blocks[:, self.triu_indices_out_i, self.triu_indices_out_j] = Q_left
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Neumann series approximation
            I = torch.eye(self.d_out, device=self.Q_left.device, dtype=torch.float32)
            R_left = I
            A = -Q_blocks
            sign = -1.0
            for i in range(1, self.num_neumann_terms + 1):
                R_left += sign * A
                A = torch.matmul(A, -Q_blocks)
                sign *= -1.0
            
            R_left = R_left.to(self.Q_left.dtype)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            Q_right = self.Q_right.to(torch.float32)
            
            # Create skew-symmetric matrix using block_num and d_in
            Q_blocks = torch.zeros(self.block_num, self.d_in, self.d_in, device=self.Q_right.device, dtype=torch.float32)
            Q_blocks[:, self.triu_indices_in_i, self.triu_indices_in_j] = Q_right
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Neumann series approximation
            I = torch.eye(self.d_in, device=self.Q_right.device, dtype=torch.float32)
            R_right = I
            A = -Q_blocks
            sign = -1.0
            for i in range(1, self.num_neumann_terms + 1):
                R_right += sign * A
                A = torch.matmul(A, -Q_blocks)
                sign *= -1.0
            
            R_right = R_right.to(self.Q_right.dtype)

        return R_left, R_right


    def get_cayley_transform_neumann_optimized(self, mode='all'):
        """
        Ultra-optimized version of get_cayley_transform_neumann.
        """
        R_left = None
        R_right = None
        
        # Pre-compute common quantities once
        if mode in ['all', 'left', 'right']:
            # Create identity matrices in advance
            I_out = torch.eye(self.d_out, device=self.Q_left.device, dtype=self.Q_left.dtype)
            I_in = torch.eye(self.d_in, device=self.Q_right.device, dtype=self.Q_right.dtype)
        
        # Process left transform if needed
        if mode in ['all', 'left']:
            # Create skew-symmetric matrix more efficiently
            # Q_blocks = torch.zeros(self.block_num, self.d_out, self.d_out, 
            #                     device=self.Q_left.device, dtype=self.Q_left.dtype)
            # Q_blocks[:, self.triu_indices_out_i, self.triu_indices_out_j] = self.Q_left
            # Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            # Q_blocks = 0.5 * (self.Q_left - self.Q_left.transpose(-2, -1))
            Q_blocks = SkewSymmetric.apply(self.Q_left, self.d_out)
            
            # Initialize result - use existing identity and expand in-place
            if self.stochastic:
                R_left = torch.eye(self.d_out, device=self.Q_left.device, dtype=self.Q_left.dtype)
            else:
                R_left = I_out.expand(self.block_num, self.d_out, self.d_out).clone()
            
            # For small matrices, unroll the first few iterations
            if self.num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_left.add_(Q_blocks, alpha=2.0)
                
                if self.num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    if self.stochastic:
                        Q_squared = Q_blocks @ Q_blocks
                    else:
                        Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_left.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, self.num_neumann_terms):
                        if self.stochastic:
                            Q_power = Q_power @ Q_blocks
                        else:
                            Q_power = torch.bmm(Q_power, Q_blocks)
                        R_left.add_(Q_power, alpha=2.0)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            # Create skew-symmetric matrix more efficiently
            # Q_blocks = torch.zeros(self.block_num, self.d_in, self.d_in, 
            #                     device=self.Q_right.device, dtype=self.Q_right.dtype)
            # Q_blocks[:, self.triu_indices_in_i, self.triu_indices_in_j] = self.Q_right
            # Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            # Q_blocks = 0.5 * (self.Q_right - self.Q_right.transpose(-2, -1))
            Q_blocks = SkewSymmetric.apply(self.Q_right, self.d_in)

            # Initialize result - use existing identity and expand in-place
            if self.stochastic:
                R_right = torch.eye(self.d_in, device=self.Q_right.device, dtype=self.Q_right.dtype)
            else:
                R_right = I_in.expand(self.block_num, self.d_in, self.d_in).clone()
            
            # For small matrices, unroll the first few iterations
            if self.num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_right.add_(Q_blocks, alpha=2.0)
                
                if self.num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    if self.stochastic:
                        Q_squared = Q_blocks @ Q_blocks
                    else:
                        Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_right.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, self.num_neumann_terms):
                        if self.stochastic: 
                            Q_power = Q_power @ Q_blocks
                        else:
                            Q_power = torch.bmm(Q_power, Q_blocks)
                        R_right.add_(Q_power, alpha=2.0)

        return R_left, R_right


    def get_cayley_transform_neumann_optimized_working_but_slow(self, mode='all'):
        """
        Ultra-optimized version of get_cayley_transform_neumann.
        """
        R_left = None
        R_right = None
        
        # Pre-compute common quantities once
        if mode in ['all', 'left', 'right']:
            # Create identity matrices in advance
            I_out = torch.eye(self.d_out, device=self.Q_left.device, dtype=self.Q_left.dtype)
            I_in = torch.eye(self.d_in, device=self.Q_right.device, dtype=self.Q_right.dtype)
        
        # Process left transform if needed
        if mode in ['all', 'left']:
            # Create skew-symmetric matrix more efficiently
            Q_blocks = torch.zeros(self.block_num, self.d_out, self.d_out, 
                                device=self.Q_left.device, dtype=self.Q_left.dtype)
            Q_blocks[:, self.triu_indices_out_i, self.triu_indices_out_j] = self.Q_left
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            # Q_blocks = 0.5 * (self.Q_left - self.Q_left.transpose(-2, -1))
            
            # Initialize result - use existing identity and expand in-place
            R_left = I_out.expand(self.block_num, self.d_out, self.d_out).clone()
            
            # For small matrices, unroll the first few iterations
            if self.num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_left.add_(Q_blocks, alpha=2.0)
                
                if self.num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_left.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, self.num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_blocks)
                        R_left.add_(Q_power, alpha=2.0)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            # Create skew-symmetric matrix more efficiently
            Q_blocks = torch.zeros(self.block_num, self.d_in, self.d_in, 
                                device=self.Q_right.device, dtype=self.Q_right.dtype)
            Q_blocks[:, self.triu_indices_in_i, self.triu_indices_in_j] = self.Q_right
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            # Q_blocks = 0.5 * (self.Q_right - self.Q_right.transpose(-2, -1))
            
            # Initialize result - use existing identity and expand in-place
            R_right = I_in.expand(self.block_num, self.d_in, self.d_in).clone()
            
            # For small matrices, unroll the first few iterations
            if self.num_neumann_terms > 1:
                # First term (i=1): Add 2*Q
                R_right.add_(Q_blocks, alpha=2.0)
                
                if self.num_neumann_terms > 2:
                    # Second term (i=2): Add 2*Q^2
                    Q_squared = torch.bmm(Q_blocks, Q_blocks)
                    R_right.add_(Q_squared, alpha=2.0)
                    
                    # Use bmm for remaining iterations
                    Q_power = Q_squared
                    for i in range(3, self.num_neumann_terms):
                        Q_power = torch.bmm(Q_power, Q_blocks)
                        R_right.add_(Q_power, alpha=2.0)

        return R_left, R_right
    

    def get_cayley_transform_neumann(self, mode='all'):
        """
        Get Cayley transform matrices using Neumann series approximation.
        """
        R_left = None
        R_right = None
        
        # Process left transform if needed
        if mode in ['all', 'left']:
            # Create skew-symmetric matrix using block_num and d_out
            Q_blocks = torch.zeros(self.block_num, self.d_out, self.d_out, device=self.Q_left.device, dtype=self.Q_left.dtype)
            Q_blocks[:, self.triu_indices_out_i, self.triu_indices_out_j] = self.Q_left
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Identity matrix with proper dimension
            I = torch.eye(self.d_out, device=self.Q_left.device, dtype=self.Q_left.dtype)
            
            # Initialize result and compute Neumann series
            R_left = I.clone().expand(self.block_num, self.d_out, self.d_out)
            Q_power = Q_blocks.clone()
            coeff = 2.0
            
            for i in range(1, self.num_neumann_terms):
                R_left = R_left + coeff * Q_power
                if i < self.num_neumann_terms - 1:
                    Q_power = torch.matmul(Q_power, Q_blocks)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            # Create skew-symmetric matrix using block_num and d_in
            Q_blocks = torch.zeros(self.block_num, self.d_in, self.d_in, device=self.Q_right.device, dtype=self.Q_right.dtype)
            Q_blocks[:, self.triu_indices_in_i, self.triu_indices_in_j] = self.Q_right
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Identity matrix with proper dimension
            I = torch.eye(self.d_in, device=self.Q_right.device, dtype=self.Q_right.dtype)
            
            # Initialize result and compute Neumann series
            R_right = I.clone().expand(self.block_num, self.d_in, self.d_in)
            Q_power = Q_blocks.clone()
            coeff = 2.0
            
            for i in range(1, self.num_neumann_terms):
                R_right = R_right + coeff * Q_power
                if i < self.num_neumann_terms - 1:
                    Q_power = torch.matmul(Q_power, Q_blocks)

        return R_left, R_right

    
    def verify_neumann_implementations(self, num_runs=5, rtol=1e-5, atol=1e-7):
        """
        Verify that the original and optimized Neumann transform implementations 
        produce the same numerical results.
        
        Args:
            num_runs (int): Number of runs to check
            rtol (float): Relative tolerance for comparison
            atol (float): Absolute tolerance for comparison
            
        Returns:
            bool: True if implementations are equivalent, False otherwise
        """
        import torch
        
        all_equal = True
        max_diff_left = 0.0
        max_diff_right = 0.0
        
        for run in range(num_runs):
            # Get results from both implementations
            R_left_orig, R_right_orig = self.get_cayley_transform_neumann()
            R_left_opt, R_right_opt = self.get_cayley_transform_neumann_optimized()
            
            # Check if results are equal within tolerance
            left_equal = torch.allclose(R_left_orig, R_left_opt, rtol=rtol, atol=atol)
            right_equal = torch.allclose(R_right_orig, R_right_opt, rtol=rtol, atol=atol)
            
            # Calculate max differences
            left_diff = torch.abs(R_left_orig - R_left_opt).max().item()
            right_diff = torch.abs(R_right_orig - R_right_opt).max().item()
            
            max_diff_left = max(max_diff_left, left_diff)
            max_diff_right = max(max_diff_right, right_diff)
            
            if not (left_equal and right_equal):
                all_equal = False
        
        # Print results
        print(f"\n{'=' * 50}")
        print(f"Neumann Transform Verification Results")
        print(f"{'=' * 50}")
        print(f"Mode: {'Stochastic' if self.stochastic else 'Regular'}, Rank: {self.rank}, Blocks: {self.block_num}")
        print(f"Neumann Terms: {self.num_neumann_terms}")
        
        if all_equal:
            print(f"PASSED: Both implementations produce identical results within tolerance")
        else:
            print(f"FAILED: Implementations produce different results")
        
        print(f"Max difference (left transform): {max_diff_left:.8e}")
        print(f"Max difference (right transform): {max_diff_right:.8e}")
        print(f"Tolerance settings: rtol={rtol}, atol={atol}")
        print(f"{'=' * 50}\n")
        
        return all_equal

    
    def merge_and_reset_R(self):
        """
        Merge and reset R matrices.
        """
        with torch.no_grad():
            # Compute transforms based on current parameters
            if self.use_neumann:
                R_left, R_right = self.get_cayley_transform_neumann_optimized()
            else:
                R_left, R_right = self.get_cayley_transform()

            # Apply transforms to weight directly
            transformed_weight = self.weight.clone()

            if self.stochastic:
                transformed_weight[self.selected_indices_left, :] = torch.matmul(R_left.squeeze(), transformed_weight[self.selected_indices_left, :])
                transformed_weight[:, self.selected_indices_right] = torch.matmul(transformed_weight[:, self.selected_indices_right], R_right.squeeze())
            else:
                # Create block diagonals and apply
                R_left_bs = torch.block_diag(*R_left)
                R_right_bs = torch.block_diag(*R_right)
                
                # Two-step matrix multiplication with appropriate dtype handling
                temp = torch.matmul(transformed_weight, R_right_bs)
                transformed_weight = torch.matmul(R_left_bs, temp)

            # Update weight buffer
            self.weight.copy_(transformed_weight)
            
            # Reset parameters after updating weight
            self.Q_left.data.zero_()
            self.Q_right.data.zero_()

            if self.stochastic:
                self.update_indices()
            


    def forward(self, x):
        # If reset_R flag is set, update weight directly and reset parameters
        if self.reset_R and self.update_reset_R_gap > 0 and \
           CayleyLinear.global_step_counter % self.update_reset_R_gap == 0 and \
           CayleyLinear.global_step_counter > 0:
            # print(f"Resetting R at global step {CayleyLinear.global_step_counter}")
            self.merge_and_reset_R()

        # Get transforms - will use cached values if parameters haven't changed
        if self.use_neumann:
            R_left, R_right = self.get_cayley_transform_neumann_optimized()
        else:
            R_left, R_right = self.get_cayley_transform()

        # Apply transforms to weight
        transformed_weight = self.weight.clone().to(dtype=x.dtype, device=x.device)

        if self.stochastic:
            transformed_weight[self.selected_indices_left, :] = torch.matmul(R_left.squeeze(), transformed_weight[self.selected_indices_left, :])
            transformed_weight[:, self.selected_indices_right] = torch.matmul(transformed_weight[:, self.selected_indices_right], R_right.squeeze())
        else:
            # Create block diagonal matrices
            R_left_bs = torch.block_diag(*R_left)
            R_right_bs = torch.block_diag(*R_right)

            temp = torch.matmul(transformed_weight, R_right_bs)
            transformed_weight = torch.matmul(R_left_bs, temp)

        # Regular linear transformation with transformed weight
        return F.linear(x, transformed_weight.squeeze(), self.bias)
    
    def benchmark_implementations(self, x, num_runs=100, warmup=10):
        """Benchmark original vs optimized implementations."""
        import time
        import torch.cuda as cuda
        
        # Ensure we're in stochastic mode
        if not self.stochastic:
            print("Benchmarking only valid in stochastic mode")
            return
        
        # Warmup runs
        for _ in range(warmup):
            _ = self.original_forward(x)
            _ = self.optimized_forward(x)
            
        # Time original implementation
        start = time.time()
        if cuda.is_available():
            cuda.synchronize()
            
        for _ in range(num_runs):
            _ = self.original_forward(x)
            
        if cuda.is_available():
            cuda.synchronize()
        original_time = time.time() - start
        
        # Time optimized implementation
        start = time.time()
        if cuda.is_available():
            cuda.synchronize()
            
        for _ in range(num_runs):
            _ = self.optimized_forward(x)
            
        if cuda.is_available():
            cuda.synchronize()
        optimized_time = time.time() - start
        
        # Calculate speedup
        speedup = original_time / optimized_time if optimized_time > 0 else 0
        
        print(f"\n{'=' * 50}")
        print(f"Stochastic Mode Forward Pass Benchmark ({num_runs} runs)")
        print(f"{'=' * 50}")
        print(f"Input shape: {x.shape}")
        print(f"Weight shape: {self.weight.shape}")
        print(f"Rank: {self.rank}, d_out: {self.d_out}, d_in: {self.d_in}")
        print(f"Original implementation: {original_time:.6f} seconds")
        print(f"Optimized implementation: {optimized_time:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify correctness
        with torch.no_grad():
            orig_out = self.original_forward(x)
            opt_out = self.optimized_forward(x)
            max_diff = torch.max(torch.abs(orig_out - opt_out)).item()
            mean_diff = torch.mean(torch.abs(orig_out - opt_out)).item()
            
            print(f"\nCorrectness Verification:")
            print(f"Max absolute difference: {max_diff:.8e}")
            print(f"Mean absolute difference: {mean_diff:.8e}")
            
            if max_diff < 1e-5:
                print("✓ Implementations are equivalent")
            else:
                print("⚠ Implementations produce different results!")
                
        print(f"{'=' * 50}\n")
        
        return {
            "original_time": original_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "max_diff": max_diff,
            "mean_diff": mean_diff
        }

    def profile_forward(self, x, num_runs=10):
        """
        Profile the forward pass to identify bottlenecks.
        
        Args:
            x (torch.Tensor): Input tensor
            num_runs (int): Number of runs to average timing over
        """
        import time
        import torch.cuda as cuda
        
        # Skip warming run
        with torch.no_grad():
            _ = self.forward(x)
        
        timings = {
            "get_transform": 0.0,
            "create_block_diag": 0.0,
            "matrix_mult": 0.0,
            "stochastic_ops": 0.0,
            "linear_transform": 0.0,
            "total": 0.0
        }
        
        for _ in range(num_runs):
            # Ensure GPU operations are completed before timing
            if cuda.is_available():
                cuda.synchronize()
            
            start_total = time.time()
            
            # Time transform calculation
            start = time.time()
            if self.use_neumann:
                R_left, R_right = self.get_cayley_transform_neumann_optimized()
            else:
                R_left, R_right = self.get_cayley_transform()
            if cuda.is_available():
                cuda.synchronize()
            timings["get_transform"] += time.time() - start
            
            transformed_weight = self.weight.detach().clone().to(dtype=x.dtype, device=x.device)
            
            if self.stochastic:
                # Time stochastic operations
                start = time.time()
                left_weights = transformed_weight[self.selected_indices_left, :]
                right_weights = transformed_weight[:, self.selected_indices_right]
                if cuda.is_available():
                    cuda.synchronize()
                timings["stochastic_ops"] += time.time() - start
                
                # Time matrix multiplications
                start = time.time()
                left_transformed = torch.matmul(R_left.squeeze(), left_weights)
                right_transformed = torch.matmul(right_weights, R_right.squeeze())
                if cuda.is_available():
                    cuda.synchronize()
                timings["matrix_mult"] += time.time() - start
                
                # More stochastic ops for updating
                start = time.time()
                result_weight = transformed_weight.clone()
                result_weight[self.selected_indices_left, :] = left_transformed
                result_weight[:, self.selected_indices_right] = right_transformed
                transformed_weight = result_weight
                if cuda.is_available():
                    cuda.synchronize()
                timings["stochastic_ops"] += time.time() - start
            else:
                # Time block diagonal creation
                start = time.time()
                R_left_bs = torch.block_diag(*R_left)
                R_right_bs = torch.block_diag(*R_right)
                if cuda.is_available():
                    cuda.synchronize()
                timings["create_block_diag"] += time.time() - start
                
                # Time matrix multiplications
                start = time.time()
                temp = torch.matmul(transformed_weight, R_right_bs)
                transformed_weight = torch.matmul(R_left_bs, temp)
                if cuda.is_available():
                    cuda.synchronize()
                timings["matrix_mult"] += time.time() - start
            
            # Time linear transformation
            start = time.time()
            result = F.linear(x, transformed_weight, self.bias)
            if cuda.is_available():
                cuda.synchronize()
            timings["linear_transform"] += time.time() - start
            
            timings["total"] += time.time() - start_total
        
        # Average the results
        for key in timings:
            timings[key] /= num_runs
            timings[key] *= 1000  # Convert to milliseconds
        
        print(f"\n{'=' * 50}")
        print(f"CayleyLinear Profile Results (avg over {num_runs} runs, ms)")
        print(f"{'=' * 50}")
        print(f"Mode: {'Stochastic' if self.stochastic else 'Regular'}, Rank: {self.rank}, Blocks: {self.block_num}")
        print(f"{'Operation':<20} {'Time (ms)':<10} {'% of Total':<10}")
        print(f"{'-' * 50}")
        
        for key, value in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            if key != "total":
                percent = (value / timings["total"]) * 100
                print(f"{key:<20} {value:<10.3f} {percent:<10.2f}%")
        
        print(f"{'-' * 50}")
        print(f"{'Total':<20} {timings['total']:<10.3f}")
        print(f"{'=' * 50}\n")
        
        return timings

    def profile_forward_sections(self, x, num_runs=10):
        """
        Profiles different sections of the forward method to identify bottlenecks.
        
        Args:
            x: Input tensor to forward pass
            num_runs: Number of runs to average timing over
        
        Returns:
            Dictionary with section names and their average execution times
        """
        import time
        
        # First run regular forward to warm up
        _ = self.forward(x)
        torch.cuda.synchronize() if x.is_cuda else None
        
        # Get device for creating tensors
        device = x.device
        
        # Clone input for consistent testing
        x_clone = x.clone()
        
        # Store timing results
        section_times = {}
        
        # Profile full forward pass as baseline
        torch.cuda.synchronize() if x.is_cuda else None
        start = time.time()
        for _ in range(num_runs):
            _ = self.forward(x_clone)
            torch.cuda.synchronize() if x.is_cuda else None
        end = time.time()
        section_times["full_forward"] = (end - start) / num_runs
        
        # Profile get_cayley_transform (or neumann version)
        torch.cuda.synchronize() if x.is_cuda else None
        start = time.time()
        for _ in range(num_runs):
            if self.use_neumann:
                _, _ = self.get_cayley_transform_neumann_optimized()
            else:
                _, _ = self.get_cayley_transform()
            torch.cuda.synchronize() if x.is_cuda else None
        end = time.time()
        section_times["cayley_transform_calculation"] = (end - start) / num_runs
        
        # Create sample R matrices for timing the application portions
        if self.use_neumann:
            R_left, R_right = self.get_cayley_transform_neumann_optimized()
        else:
            R_left, R_right = self.get_cayley_transform()
            
        # Profile stochastic vs non-stochastic transform application
        transformed_weight = self.weight.clone()
        
        if self.stochastic:
            # Profile stochastic transform application
            torch.cuda.synchronize() if x.is_cuda else None
            start = time.time()
            for _ in range(num_runs):
                tmp_weight = transformed_weight.clone()
                tmp_weight[self.selected_indices_left, :] = torch.matmul(
                    R_left.squeeze(), tmp_weight[self.selected_indices_left, :]
                )
                tmp_weight[:, self.selected_indices_right] = torch.matmul(
                    tmp_weight[:, self.selected_indices_right], R_right.squeeze()
                )
                torch.cuda.synchronize() if x.is_cuda else None
            end = time.time()
            section_times["stochastic_transform_application"] = (end - start) / num_runs
        else:
            # Profile block diagonal creation
            torch.cuda.synchronize() if x.is_cuda else None
            start = time.time()
            for _ in range(num_runs):
                R_left_bs = torch.block_diag(*R_left)
                R_right_bs = torch.block_diag(*R_right)
                torch.cuda.synchronize() if x.is_cuda else None
            end = time.time()
            section_times["block_diagonal_creation"] = (end - start) / num_runs
            
            # Profile matrix multiplication step 1 (weight @ R_right_bs)
            torch.cuda.synchronize() if x.is_cuda else None
            start = time.time()
            for _ in range(num_runs):
                R_left_bs = torch.block_diag(*R_left)
                R_right_bs = torch.block_diag(*R_right)
                temp = torch.matmul(transformed_weight, R_right_bs)
                torch.cuda.synchronize() if x.is_cuda else None
            end = time.time()
            section_times["matrix_mult_step1"] = (end - start) / num_runs - section_times["block_diagonal_creation"]
            
            # Profile matrix multiplication step 2 (R_left_bs @ temp)
            torch.cuda.synchronize() if x.is_cuda else None
            start = time.time()
            for _ in range(num_runs):
                R_left_bs = torch.block_diag(*R_left)
                R_right_bs = torch.block_diag(*R_right)
                temp = torch.matmul(transformed_weight, R_right_bs)
                result = torch.matmul(R_left_bs, temp)
                torch.cuda.synchronize() if x.is_cuda else None
            end = time.time()
            section_times["matrix_mult_step2"] = (end - start) / num_runs - section_times["block_diagonal_creation"] - section_times["matrix_mult_step1"]
            
        # Profile the final linear operation (without transform)
        torch.cuda.synchronize() if x.is_cuda else None
        start = time.time()
        for _ in range(num_runs):
            result = F.linear(x_clone, self.weight, self.bias)
            torch.cuda.synchronize() if x.is_cuda else None
        end = time.time()
        section_times["linear_operation"] = (end - start) / num_runs
        
        # Print results in descending order of time
        print("\n==== Forward Method Section Timing ====")
        for section, avg_time in sorted(section_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{section}: {avg_time*1000:.4f} ms")
        print("=====================================\n")
        
        return section_times

    def benchmark_forward_pass(self, x, num_runs=100, warmup=10):
        """
        Detailed benchmarking of the forward pass operations.
        
        Args:
            x (torch.Tensor): Input tensor with appropriate shape for the layer
            num_runs (int): Number of runs to average timing over
            warmup (int): Number of warmup runs before timing
            
        Returns:
            dict: Dictionary with timing results for each operation
        """
        import time
        import torch.cuda as cuda
        
        # Ensure CUDA is synchronized if using GPU
        is_cuda = x.is_cuda
        sync_cuda = lambda: cuda.synchronize() if is_cuda else None
        
        # Dictionary to store timing results
        timings = {
            "get_transform": 0.0,
            "weight_clone": 0.0,
            "block_diag_creation": 0.0,
            "first_matmul": 0.0,
            "second_matmul": 0.0,
            "stochastic_indexing": 0.0,
            "stochastic_matmul": 0.0,
            "linear_operation": 0.0,
            "total": 0.0
        }
        
        # Warmup runs
        for _ in range(warmup):
            _ = self.forward(x)
            sync_cuda()
        
        # Main benchmarking loop
        for _ in range(num_runs):
            # Time the entire forward pass for reference
            sync_cuda()
            start_total = time.time()
            
            # 1. Time transform calculation
            sync_cuda()
            start = time.time()
            if self.use_neumann:
                R_left, R_right = self.get_cayley_transform_neumann_optimized()
            else:
                R_left, R_right = self.get_cayley_transform()
            sync_cuda()
            timings["get_transform"] += time.time() - start
            
            # 2. Time weight cloning
            sync_cuda()
            start = time.time()
            transformed_weight = self.weight.clone().to(dtype=x.dtype, device=x.device)
            sync_cuda()
            timings["weight_clone"] += time.time() - start
            
            if self.stochastic:
                # 3a. Time stochastic indexing
                sync_cuda()
                start = time.time()
                left_weights = transformed_weight[self.selected_indices_left, :]
                right_weights = transformed_weight[:, self.selected_indices_right]
                sync_cuda()
                timings["stochastic_indexing"] += time.time() - start
                
                # 4a. Time stochastic matrix multiplications
                sync_cuda()
                start = time.time()
                left_transformed = torch.matmul(R_left.squeeze(), left_weights)
                right_transformed = torch.matmul(right_weights, R_right.squeeze())
                sync_cuda()
                timings["stochastic_matmul"] += time.time() - start
                
                # Update the transformed weight
                transformed_weight[self.selected_indices_left, :] = left_transformed
                transformed_weight[:, self.selected_indices_right] = right_transformed
            else:
                # 3b. Time block diagonal creation
                sync_cuda()
                start = time.time()
                R_left_bs = torch.block_diag(*R_left)
                R_right_bs = torch.block_diag(*R_right)
                sync_cuda()
                timings["block_diag_creation"] += time.time() - start
                
                # 4b. Time first matrix multiplication
                sync_cuda()
                start = time.time()
                temp = torch.matmul(transformed_weight, R_right_bs)
                sync_cuda()
                timings["first_matmul"] += time.time() - start
                
                # 5b. Time second matrix multiplication
                sync_cuda()
                start = time.time()
                transformed_weight = torch.matmul(R_left_bs, temp)
                sync_cuda()
                timings["second_matmul"] += time.time() - start
            
            # 6. Time linear transformation
            sync_cuda()
            start = time.time()
            result = F.linear(x, transformed_weight, self.bias)
            sync_cuda()
            timings["linear_operation"] += time.time() - start
            
            timings["total"] += time.time() - start_total
        
        # Average the results
        for key in timings:
            timings[key] /= num_runs
            timings[key] *= 1000  # Convert to milliseconds
        
        # Print results in a nice format
        print(f"\n{'=' * 60}")
        print(f"CayleyLinear Forward Pass Benchmark ({num_runs} runs)")
        print(f"{'=' * 60}")
        print(f"Mode: {'Stochastic' if self.stochastic else 'Regular'}")
        print(f"Input shape: {x.shape}")
        print(f"Weight shape: {self.weight.shape}")
        print(f"Rank: {self.rank}, d_out: {self.d_out}, d_in: {self.d_in}")
        print(f"Using Neumann: {self.use_neumann}")
        if self.use_neumann:
            print(f"Neumann terms: {self.num_neumann_terms}")
        print(f"{'-' * 60}")
        
        # Print operations in descending order of time
        print(f"{'Operation':<25} {'Time (ms)':<12} {'% of Total':<10}")
        print(f"{'-' * 60}")
        
        for key, value in sorted(
            [(k, v) for k, v in timings.items() if k != "total"], 
            key=lambda x: x[1], 
            reverse=True
        ):
            percent = (value / timings["total"]) * 100
            print(f"{key:<25} {value:<12.3f} {percent:<10.2f}%")
        
        print(f"{'-' * 60}")
        print(f"{'Total':<25} {timings['total']:<12.3f} 100.00%")
        print(f"{'=' * 60}\n")
        
        return timings

    def benchmark_cayley_transform(self, num_runs=100, warmup=10):
        """
        Detailed benchmarking of the Cayley transform calculation.
        
        Args:
            num_runs (int): Number of runs to average timing over
            warmup (int): Number of warmup runs before timing
            
        Returns:
            dict: Dictionary with timing results for each operation
        """
        import time
        import torch.cuda as cuda
        
        # Ensure CUDA is synchronized if using GPU
        device = next(self.parameters()).device
        is_cuda = device.type == 'cuda'
        sync_cuda = lambda: cuda.synchronize() if is_cuda else None
        
        # Dictionary to store timing results
        timings = {
            "skew_matrix_creation": 0.0,
            "matrix_solve": 0.0,
            "neumann_series": 0.0,
            "total": 0.0
        }
        
        # Warmup runs
        for _ in range(warmup):
            if self.use_neumann:
                _ = self.get_cayley_transform_neumann_optimized()
            else:
                _ = self.get_cayley_transform()
            sync_cuda()
        
        # Main benchmarking loop
        for _ in range(num_runs):
            sync_cuda()
            start_total = time.time()
            
            if self.use_neumann:
                # Benchmark Neumann series approach
                
                # 1. Time skew-symmetric matrix creation
                sync_cuda()
                start = time.time()
                Q_blocks_left = SkewSymmetric.apply(self.Q_left, self.d_out).unsqueeze(0)
                Q_blocks_right = SkewSymmetric.apply(self.Q_right, self.d_in).unsqueeze(0)
                sync_cuda()
                timings["skew_matrix_creation"] += time.time() - start
                
                # 2. Time Neumann series calculation
                sync_cuda()
                start = time.time()
                I_out = torch.eye(self.d_out, device=self.Q_left.device, dtype=torch.bfloat16)
                R_left = I_out.expand(self.block_num, self.d_out, self.d_out).clone()
                
                if self.num_neumann_terms > 1:
                    R_left.add_(Q_blocks_left, alpha=2.0)
                    
                    if self.num_neumann_terms > 2:
                        Q_squared = torch.bmm(Q_blocks_left, Q_blocks_left)
                        R_left.add_(Q_squared, alpha=2.0)
                        
                        # Optimization 1: Pre-compute powers in parallel instead of sequential loop
                        if self.num_neumann_terms > 3:
                            # Initialize list to store powers
                            Q_powers = [Q_blocks_left, Q_squared]
                            
                            # Pre-compute all required powers
                            for i in range(3, self.num_neumann_terms):
                                Q_powers.append(torch.bmm(Q_powers[-1], Q_blocks_left))
                            
                            # Add all powers to R_left at once
                            for i in range(2, self.num_neumann_terms - 1):
                                R_left.add_(Q_powers[i], alpha=2.0)
                        
                        # Original sequential approach
                        # Q_power = Q_squared
                        # for i in range(3, self.num_neumann_terms):
                        #     Q_power = torch.bmm(Q_power, Q_blocks_left)
                        #     R_left.add_(Q_power, alpha=2.0)
                
                # Repeat for right transform
                I_in = torch.eye(self.d_in, device=self.Q_right.device, dtype=torch.bfloat16)
                R_right = I_in.expand(self.block_num, self.d_in, self.d_in).clone()
                
                if self.num_neumann_terms > 1:
                    R_right.add_(Q_blocks_right, alpha=2.0)
                    
                    if self.num_neumann_terms > 2:
                        Q_squared = torch.bmm(Q_blocks_right, Q_blocks_right)
                        R_right.add_(Q_squared, alpha=2.0)
                        
                        # Optimization 1: Pre-compute powers in parallel instead of sequential loop
                        if self.num_neumann_terms > 3:
                            # Initialize list to store powers
                            Q_powers = [Q_blocks_right, Q_squared]
                            
                            # Pre-compute all required powers
                            for i in range(3, self.num_neumann_terms):
                                Q_powers.append(torch.bmm(Q_powers[-1], Q_blocks_right))
                            
                            # Add all powers to R_right at once
                            for i in range(2, self.num_neumann_terms - 1):
                                R_right.add_(Q_powers[i], alpha=2.0)
                        
                        # Original sequential approach
                        # Q_power = Q_squared
                        # for i in range(3, self.num_neumann_terms):
                        #     Q_power = torch.bmm(Q_power, Q_blocks_right)
                        #     R_right.add_(Q_power, alpha=2.0)
                sync_cuda()
                timings["neumann_series"] += time.time() - start
                
            else:
                # Benchmark standard Cayley transform
                
                # 1. Time skew-symmetric matrix creation
                sync_cuda()
                start = time.time()
                Q_left = self.Q_left.to(torch.bfloat16)
                Q_blocks_left = torch.zeros(self.block_num, self.d_out, self.d_out, 
                                        device=self.Q_left.device, dtype=torch.bfloat16)
                Q_blocks_left[:, self.triu_indices_out_i, self.triu_indices_out_j] = Q_left
                Q_blocks_left = Q_blocks_left - Q_blocks_left.transpose(-2, -1)
                
                Q_right = self.Q_right.to(torch.bfloat16)
                Q_blocks_right = torch.zeros(self.block_num, self.d_in, self.d_in, 
                                        device=self.Q_right.device, dtype=torch.bfloat16)
                Q_blocks_right[:, self.triu_indices_in_i, self.triu_indices_in_j] = Q_right
                Q_blocks_right = Q_blocks_right - Q_blocks_right.transpose(-2, -1)
                sync_cuda()
                timings["skew_matrix_creation"] += time.time() - start
                
                # 2. Time matrix solve operation
                sync_cuda()
                start = time.time()
                I_out = torch.eye(self.d_out, device=self.Q_left.device, dtype=torch.bfloat16)
                R_left = torch.linalg.solve(I_out - Q_blocks_left, I_out + Q_blocks_left).transpose(-2, -1)
                
                I_in = torch.eye(self.d_in, device=self.Q_right.device, dtype=torch.bfloat16)
                R_right = torch.linalg.solve(I_in - Q_blocks_right, I_in + Q_blocks_right).transpose(-2, -1)
                sync_cuda()
                timings["matrix_solve"] += time.time() - start
            
            timings["total"] += time.time() - start_total
        
        # Average the results
        for key in timings:
            timings[key] /= num_runs
            timings[key] *= 1000  # Convert to milliseconds
        
        # Print results in a nice format
        print(f"\n{'=' * 60}")
        print(f"Cayley Transform Calculation Benchmark ({num_runs} runs)")
        print(f"{'=' * 60}")
        print(f"Mode: {'Neumann Series' if self.use_neumann else 'Standard Cayley'}")
        if self.use_neumann:
            print(f"Neumann terms: {self.num_neumann_terms}")
        print(f"Block number: {self.block_num}")
        print(f"Matrix dimensions: d_out={self.d_out}, d_in={self.d_in}")
        print(f"{'-' * 60}")
        
        # Print operations in descending order of time
        print(f"{'Operation':<25} {'Time (ms)':<12} {'% of Total':<10}")
        print(f"{'-' * 60}")
        
        for key, value in sorted(
            [(k, v) for k, v in timings.items() if k != "total"], 
            key=lambda x: x[1], 
            reverse=True
        ):
            percent = (value / timings["total"]) * 100
            print(f"{key:<25} {value:<12.3f} {percent:<10.2f}%")
        
        print(f"{'-' * 60}")
        print(f"{'Total':<25} {timings['total']:<12.3f} 100.00%")
        print(f"{'=' * 60}\n")
        
        return timings


def load_local_data(split='train', max_samples=None, seed=42):
    """
    Load local C4 data with reproducible shuffling, loading files one by one until
    reaching the desired number of samples.
    
    Args:
        split: 'train' or 'validation'
        max_samples: Maximum number of samples to load (None for all)
        seed: Random seed for reproducible shuffling
    """
    features = Features({
        'text': Value('string'),
        'timestamp': Value('string'),
        'url': Value('string')
    })
    
    data_dir = "./c4/en"
    import glob

    # Determine cache directory
    if os.path.exists("/local/reservation"):
        cache_dir = "/local/reservation/c4"
    else:
        cache_dir = "/tmp/c4"
    
    # Get all available files
    all_files = sorted(glob.glob(os.path.join(data_dir, f"c4-{split}.*.json.gz")))
    
    if not all_files:
        raise ValueError(f"No files found in {data_dir} matching c4-{split}.*.json.gz")
    
    # Use deterministic file order based on seed
    random.seed(seed)
    random.shuffle(all_files)
    
    # For validation split, load all files regardless of max_samples
    if split == 'validation':
        max_samples = None
    
    # Load files one by one until we have enough samples
    collected_datasets = []
    total_samples = 0
    files_used = 0
    
    for file_path in all_files:
        try:
            # Explicitly disable caching for all dataset operations
            file_dataset = datasets.load_dataset(
                "json",
                data_files=file_path,
                features=features,
                streaming=False,
                cache_dir=cache_dir,               # Explicitly disable cache
                # keep_in_memory=True,       # Keep everything in memory
                num_proc=32,
                download_config=DownloadConfig(cache_dir=None, force_download=True)  # Additional cache disabling
            )
            
            file_samples = len(file_dataset['train'])
            files_used += 1
            
            # Add to our collection
            collected_datasets.append(file_dataset['train'])
            total_samples += file_samples
            
            logger.info(f"Loaded file {files_used}: {file_path} with {file_samples} samples. Total: {total_samples}")
            
            # Check if we have enough samples (only for train split)
            if max_samples is not None and total_samples >= max_samples:
                logger.info(f"Reached target of {max_samples} samples after loading {files_used} files")
                break
                
        except Exception as e:
            logger.warning(f"Error loading file {file_path}: {e}. Skipping.")
    
    # Combine all loaded datasets
    if collected_datasets:
        combined_dataset = datasets.concatenate_datasets(collected_datasets)
        
        # Shuffle the combined dataset
        combined_dataset = combined_dataset.shuffle(seed=seed)
        
        # Take exactly max_samples if we have more (only for train split)
        # if max_samples is not None and len(combined_dataset) > max_samples:
        #    combined_dataset = combined_dataset.select(range(max_samples))
            
        logger.info(f"Final dataset has {len(combined_dataset)} samples from {files_used} files")
        return combined_dataset
    else:
        raise ValueError("Failed to load any valid files")


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts", "constant"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--data_streaming", default=False, action="store_true")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # SOFT parameters
    parser.add_argument("--soft_lr", type=float, default=1e-4)
    parser.add_argument("--soft_rank", type=int, default=128)
    parser.add_argument("--soft_num_neumann_terms", type=int, default=1)
    parser.add_argument("--soft_scale", type=float, default=1.0)
    parser.add_argument("--soft_proj_type", type=str, default="std")
    parser.add_argument("--reset_R", default=False, action="store_true")
    parser.add_argument("--update_reset_R_gap", type=int, default=50)
    parser.add_argument("--stochastic", default=False, action="store_true")

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    #wandb
    parser.add_argument("--wandb_project", type=str, default="soft-c4")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    # Val data loading with explicit cache disabling
    val_data = load_local_data(split='validation', seed=42)  # Using our modified function
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    # Create a non-caching version of preprocessing
    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
        # cache_dir="/tmp/c4",
        num_proc=16,
        # cache_file_name=None,  # Disable caching during preprocessing
        load_from_cache_file=True
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size
    
    # Calculate perplexity
    perplexity = math.exp(total_loss)

    return total_loss, perplexity, evaluated_on_tokens


def benchmark_model_cayley_layers(model, device, batch_size=32, seq_length=128):
    """
    Run comprehensive benchmarks on all CayleyLinear layers in a model.
    
    Args:
        model: The neural network model
        device: Device to run benchmarks on
        batch_size: Batch size for sample inputs
        seq_length: Sequence length for sample inputs (for transformer models)
        
    Returns:
        dict: Dictionary with benchmark results for each layer
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE CAYLEY LAYER BENCHMARKING")
    print("="*80)
    
    # Find all CayleyLinear layers
    cayley_layers = []
    for name, module in model.named_modules():
        if isinstance(module, CayleyLinear):
            cayley_layers.append((name, module))
    
    if not cayley_layers:
        print("No CayleyLinear layers found in model")
        return {}
    
    print(f"Found {len(cayley_layers)} CayleyLinear layers")
    
    # Store results for each layer
    results = {}
    
    # Benchmark each layer
    for i, (name, layer) in enumerate(cayley_layers):
        print(f"\n{'-'*80}")
        print(f"Benchmarking layer {i+1}/{len(cayley_layers)}: {name}")
        print(f"{'-'*80}")
        
        # Create appropriate input size for this layer
        in_features = layer.in_features
        
        # For attention layers in transformers, the input might be [batch, seq_len, hidden_dim]
        if "attn" in name.lower():
            sample_input = torch.randn(batch_size, seq_length, in_features, 
                                      device=device, dtype=torch.bfloat16)
            # Reshape to [batch*seq_len, hidden_dim] for the linear layer
            sample_input = sample_input.reshape(-1, in_features)
        else:
            sample_input = torch.randn(batch_size, in_features, 
                                      device=device, dtype=torch.bfloat16)
        
        # Run the benchmarks
        forward_results = layer.benchmark_forward_pass(sample_input, num_runs=50, warmup=10)
        transform_results = layer.benchmark_cayley_transform(num_runs=50, warmup=10)
        
        # Store results
        results[name] = {
            "forward": forward_results,
            "transform": transform_results,
            "layer_info": {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "rank": layer.rank,
                "stochastic": layer.stochastic,
                "use_neumann": layer.use_neumann,
                "num_neumann_terms": getattr(layer, "num_neumann_terms", 0)
            }
        }
    
    print("\n" + "="*80)
    print("BENCHMARKING SUMMARY")
    print("="*80)
    
    # Print summary of slowest operations across all layers
    all_forward_ops = {}
    for name, result in results.items():
        for op, time in result["forward"].items():
            if op != "total":
                if op not in all_forward_ops:
                    all_forward_ops[op] = []
                all_forward_ops[op].append((name, time))
    
    print("\nSlowest operations across all layers:")
    for op, times in all_forward_ops.items():
        avg_time = sum(t for _, t in times) / len(times)
        max_time = max(times, key=lambda x: x[1])
        print(f"{op:<25}: Avg {avg_time:.3f} ms, Max {max_time[1]:.3f} ms in {max_time[0]}")
    
    print("\n" + "="*80)
    
    return results


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project=args.wandb_project, name=args.save_dir.split('/')[-1])
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # Calculate how many samples we need based on training steps
    samples_needed = args.num_training_steps * args.total_batch_size
    # Add some buffer (20%) to account for filtering, etc.
    samples_needed = int(samples_needed * 1.2)
    logger.info(f"Auto-calculated samples needed: {samples_needed} based on {args.num_training_steps} steps")
    
    seed_for_shuffle = args.seed  # Use the same seed as the rest of the training
    
    # Try to load local data with limited samples
    local_data = load_local_data(split='train', max_samples=samples_needed, seed=seed_for_shuffle)
    
    if local_data is not None:
        # Use the local data we loaded
        data = local_data
    else:
        # Fall back to original streaming approach
        logger.info("Using original streaming approach")
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
        data = data.shuffle(seed=seed_for_shuffle)
    
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    # Store initial model structure
    initial_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            initial_layers[name] = type(module).__name__

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)


    if 'galore' in args.optimizer.lower() or args.optimizer.lower() == "only_adamw":
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        galore_layer_names = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            # print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
            galore_layer_names.append(module_name)
        
        print(f"GaLore will be applied to {len(galore_params)} layers")

        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        if not 'only' in args.optimizer.lower():
            param_groups = [
                {'params': regular_params}, 
                {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}
            ]
        else:
            param_groups = [
                {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}
            ]

        # num_regular_params = 0
        # for p in regular_params:
        #     num_regular_params += p.numel()
        # print(f"Number of regular parameters (before GaLore): {num_regular_params}")


    if 'soft' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        soft_params = []  # for storing Q parameters
        scale_params = []  # for storing scale parameters
        soft_layer_names = []
        regular_params = []  # for storing other parameters
        target_modules_list = ["attn", "mlp"]
        
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

             # Store the original neuron norms before replacing the layer
            with torch.no_grad():
                # Normalize the original weights
                module.weight.data = module.weight.data / torch.norm(module.weight.data, dim=1, keepdim=True)
            
            # print('enable Cayley transform for weights in module: ', module_name)
            # Replace the linear layer with CayleyLinear
            parent_name, child_name = module_name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)
            new_layer = CayleyLinear(module, rank=args.soft_rank, stochastic=args.stochastic)
            new_layer.reset_R = args.reset_R
            new_layer.update_reset_R_gap = args.update_reset_R_gap
            if "neumann" in args.optimizer.lower():
                new_layer.use_neumann = True
                new_layer.num_neumann_terms = args.soft_num_neumann_terms
            setattr(parent_module, child_name, new_layer)

            # Add Q parameter to soft_params
            soft_params.append(new_layer.Q_left)
            soft_params.append(new_layer.Q_right)
            # scale_params.append(new_layer.scale)
            soft_layer_names.append(module_name)
            # if new_layer.bias is not None:
            #     regular_params.append(new_layer.bias)
        print(f"SOFT will be applied to {len(soft_layer_names)} layers")
        
        id_soft_params = [id(p) for p in soft_params]
        # Add all other parameters to regular_params
        regular_params = [p for p in model.parameters() if id(p) not in id_soft_params]
        # Create parameter groups for optimizer
        if not 'only' in args.optimizer.lower():
            param_groups = [
                {'params': regular_params, 'lr': args.lr},
                {'params': soft_params, 'lr': args.soft_lr, 'soft_rank': args.soft_rank, 'update_reset_R_gap': args.update_reset_R_gap, 'stochastic': args.stochastic},
            ]
        else:
            param_groups = [
                {'params': soft_params, 'lr': args.soft_lr, 'soft_rank': args.soft_rank, 'update_reset_R_gap': args.update_reset_R_gap, 'stochastic': args.stochastic},
            ]

        # num_regular_params = 0
        # for p in regular_params:
        #     num_regular_params += p.numel()
        # print(f"Number of regular parameters (before SOFT): {num_regular_params}")


    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    if 'soft' in args.optimizer.lower():
        logger.info(f"Total params with SOFT enabled: {sum(p.numel() for p in soft_params) / 1_000_000:.2f}M")


    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "only_adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw" or args.optimizer.lower() == "only_galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "soft_adamw_neumann" or args.optimizer.lower() == "only_soft_adamw_neumann" or args.optimizer.lower() == "only_soft_adamw" or args.optimizer.lower() == "soft_adamw":
        # redefine way to call soft_adamw
        if args.stochastic:
            optimizer = SOFTAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    # implement sgd
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    # implement adafactor
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # low-rank adafactor
    elif args.optimizer.lower() == "galore_adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    # 8-bit Adam
    elif args.optimizer.lower() == "adam8bit":
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw8bit":
        optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'galore_adamw8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap
        optimizer_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args.rank, 'update_proj_gap': args.update_proj_gap * 2, 'scale': args.galore_scale, 'proj_type': args.proj_type}], lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args.lr, weight_decay=args.weight_decay)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheduler(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * 2,
                    warmup_steps=args.warmup_steps * 2,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
                
        layer_wise_flag = True
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheduler(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # ##############################
    # Memory Usage Analysis
    # ##############################
    if global_rank == 0:
        memory_usage = memory_utils.calculate_memory_usage(
            model=model.module if hasattr(model, 'module') else model,
            optimizer_name=args.optimizer,
            dtype=model.dtype if hasattr(model, 'dtype') else model.module.dtype,
            trainable_params=trainable_params if not any(x in args.optimizer.lower() for x in ['galore', 'soft']) else None,  # For regular optimizers
            param_groups=param_groups if any(x in args.optimizer.lower() for x in ['galore', 'soft']) else None,  # For GaLore/SOFT
            rank=args.rank if 'galore' in args.optimizer.lower() else args.soft_rank if 'soft' in args.optimizer.lower() else None
        )
        
        logger.info("Theoretical Memory Usage Analysis:")
        logger.info(f"Parameter Memory    : {memory_usage['parameter_size_gb']:.6f} GB")
        logger.info(f"Optimizer Memory    : {memory_usage['optimizer_size_gb']:.6f} GB")
        logger.info(f"GaLore Extra Memory : {memory_usage['galore_size_gb']:.6f} GB")
        logger.info(f"SOFT Extra Memory   : {memory_usage['soft_size_gb']:.6f} GB")
        logger.info(f"Total Memory        : {memory_usage['total_size_gb']:.6f} GB")
        logger.info(f"Parameter Count     : {memory_usage['param_count_m']:.6f}M")

        # Log memory usage to wandb
        wandb.run.summary.update({
            "memory/parameter_size_gb": f"{memory_usage['parameter_size_gb']:.6f} GB",
            "memory/optimizer_size_gb": f"{memory_usage['optimizer_size_gb']:.6f} GB",
            "memory/galore_size_gb": f"{memory_usage['galore_size_gb']:.6f} GB",
            "memory/soft_size_gb": f"{memory_usage['soft_size_gb']:.6f} GB",
            "memory/total_size_gb": f"{memory_usage['total_size_gb']:.6f} GB",
            "memory/param_count_m": f"{memory_usage['param_count_m']:.6f}M"
        })

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    import time
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    '''
    # Add this to your main function after model initialization
    if global_rank == 0:
        print("\nRunning benchmarks on CayleyLinear layers...")
        benchmark_model_cayley_layers(
            model.module if hasattr(model, 'module') else model,
            device,
            batch_size=args.batch_size,
            seq_length=args.max_length
        )
    '''

    for batch_idx, batch in enumerate(dataloader):
        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # Fix gradient clipping for parameter groups
        if 'soft' in args.optimizer.lower() or 'galore' in args.optimizer.lower():
            if args.grad_clipping != 0.0:
                # Extract parameters from parameter groups
                parameters = []
                for group in param_groups:
                    parameters.extend(group['params'])
                
                # Apply gradient clipping to the flattened list of parameters
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clipping)
        else:
            if args.grad_clipping != 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Increment the global step counter for CayleyLinear layers
            CayleyLinear.global_step_counter += 1

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            
            # Set pad_token_id in generation config to avoid validation errors
            if hasattr(model, 'module'):
                model.module.generation_config.pad_token_id = tokenizer.pad_token_id
                model.module.save_pretrained(current_model_directory)
            else:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
                model.save_pretrained(current_model_directory)
            
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)
                
            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, perplexity, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_perplexity": perplexity,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}, perplexity: {perplexity:.2f}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                },
                step=global_step,
            )
        update_time = time.time()


        '''
        # Add this where you want to compare the Neumann transform implementations
        if global_rank == 0 and update_step == 20:  # Profile after 20 update steps
            with torch.no_grad():  # Disable gradients for verification
                for name, module in model.named_modules():
                    if isinstance(module, CayleyLinear):
                        print(f"\nVerifying Neumann transform implementations for layer: {name}")
                        # Set up Neumann parameters
                        module.use_neumann = True
                        
                        # Try with different numbers of terms
                        for terms in [3, 5, 10]:
                            module.num_neumann_terms = terms
                            print(f"\nTesting with {terms} Neumann terms:")
                            
                            # Test in both regular and stochastic modes
                            for stochastic in [False]:
                                module.stochastic = stochastic
                                verification_result = module.verify_neumann_implementations()
                                
                                # If verification fails, you might want to stop and fix the issue
                                if not verification_result:
                                    breakpoint()
                                    print("WARNING: Implementations are not equivalent! Fix before using in production.")
                        
                        # Performance profiling
                        print("\n" + "="*80)
                        print(f"PERFORMANCE PROFILING FOR LAYER: {name}")
                        print("="*80)
                        
                        # Profile with different term counts to see how performance scales
                        for terms in [3, 5, 10, 15]:
                            module.num_neumann_terms = terms
                            
                            # Set up timing
                            import time
                            num_runs = 100
                            warmup_runs = 10
                            times = {"original": 0.0, "optimized": 0.0, "optimized2": 0.0}
                            
                            # Warmup runs
                            for _ in range(warmup_runs):
                                _ = module.get_cayley_transform_neumann()
                                _ = module.get_cayley_transform_neumann_optimized()
                                if hasattr(module, 'get_cayley_transform_neumann_optimized2'):
                                    _ = module.get_cayley_transform_neumann_optimized2()
                            
                            # Force CUDA synchronization if using GPU
                            torch.cuda.synchronize() if next(module.parameters()).is_cuda else None
                            
                            # Benchmark original implementation
                            start = time.time()
                            for _ in range(num_runs):
                                _ = module.get_cayley_transform_neumann()
                                torch.cuda.synchronize() if next(module.parameters()).is_cuda else None
                            times["original"] = (time.time() - start) / num_runs * 1000  # Convert to ms
                            
                            # Store result for correctness comparison
                            original_result = module.get_cayley_transform_neumann()
                            
                            # Benchmark optimized implementation
                            start = time.time()
                            for _ in range(num_runs):
                                _ = module.get_cayley_transform_neumann_optimized()
                                torch.cuda.synchronize() if next(module.parameters()).is_cuda else None
                            times["optimized"] = (time.time() - start) / num_runs * 1000  # Convert to ms
                            
                            # Store result for correctness comparison
                            optimized_result = module.get_cayley_transform_neumann_optimized()
                            
                            # Benchmark optimized2 implementation if available
                            if hasattr(module, 'get_cayley_transform_neumann_optimized2'):
                                start = time.time()
                                for _ in range(num_runs):
                                    _ = module.get_cayley_transform_neumann_optimized2()
                                    torch.cuda.synchronize() if next(module.parameters()).is_cuda else None
                                times["optimized2"] = (time.time() - start) / num_runs * 1000  # Convert to ms
                                
                                # Store result for correctness comparison
                                optimized2_result = module.get_cayley_transform_neumann_optimized2()
                            
                            # Print results
                            print(f"\nBenchmark results with {terms} Neumann terms (average over {num_runs} runs):")
                            print(f"Original:   {times['original']:.4f} ms")
                            print(f"Optimized:  {times['optimized']:.4f} ms")
                            if hasattr(module, 'get_cayley_transform_neumann_optimized2'):
                                print(f"Optimized2: {times['optimized2']:.4f} ms")
                            
                            # Print speedups
                            if times["original"] > 0:
                                print(f"Optimized speedup:  {times['original']/times['optimized']:.2f}x")
                                if hasattr(module, 'get_cayley_transform_neumann_optimized2') and times["optimized2"] > 0:
                                    print(f"Optimized2 speedup: {times['original']/times['optimized2']:.2f}x")
                                    print(f"Optimized2 vs Optimized: {times['optimized']/times['optimized2']:.2f}x")
                            
                            # Check numerical accuracy
                            if hasattr(module, 'get_cayley_transform_neumann_optimized2'):
                                # Compare Optimized vs Original
                                def max_relative_diff(a, b):
                                    if a is None or b is None:
                                        return float('inf')
                                    if isinstance(a, tuple) and isinstance(b, tuple):
                                        return max(max_relative_diff(a[0], b[0]) if a[0] is not None else 0,
                                                 max_relative_diff(a[1], b[1]) if a[1] is not None else 0)
                                    if a is None or b is None:
                                        return 0
                                    diff = torch.abs(a - b)
                                    scale = torch.max(torch.abs(a), torch.abs(b))
                                    rel_diff = torch.where(scale > 1e-7, diff / scale, diff)
                                    return rel_diff.max().item() if rel_diff.numel() > 0 else 0
                                
                                opt_vs_orig = max_relative_diff(optimized_result, original_result)
                                opt2_vs_orig = max_relative_diff(optimized2_result, original_result)
                                opt2_vs_opt = max_relative_diff(optimized2_result, optimized_result)
                                
                                print("\nNumerical accuracy (max relative difference):")
                                print(f"Optimized vs Original:  {opt_vs_orig:.2e}")
                                print(f"Optimized2 vs Original: {opt2_vs_orig:.2e}")
                                print(f"Optimized2 vs Optimized: {opt2_vs_opt:.2e}")
                            
                        print("="*80)

            '''

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Set pad_token_id in generation config to avoid validation errors
        if hasattr(model, 'module'):
            model.module.generation_config.pad_token_id = tokenizer.pad_token_id
            model.module.save_pretrained(current_model_directory)
        else:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, perplexity, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_perplexity": perplexity,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}, perplexity: {perplexity:.2f}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)

