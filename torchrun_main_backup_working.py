import os
import time
import json
import random
import argparse
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist

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
from soft_torch import matmul

transformers.logging.set_verbosity_error()

import math
import warnings
from typing import Callable, Iterable, Optional, Tuple, Union
class SOFTAdamW(torch.optim.Optimizer):
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
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
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

                # if torch.all(p == 0) and len(state) != 0 and state["step"] != 0:
                #     state["step"] = 0
                #     state["exp_avg"] = torch.zeros_like(p)
                #     state["exp_avg_sq"] = torch.zeros_like(p)
                #     continue

                # State initialization
                if len(state) == 0 or torch.all(p == 0):
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
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


class CayleyLinear(nn.Linear):
    def __init__(self, linear_layer, rank=128):
        """Initialize CayleyLinear by inheriting from an existing linear layer."""
        super().__init__(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None
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
        
        # Store rank and initialize Q matrix with reduced size on the same device and dtype as weight
        # self.rank = min(rank, self.in_features)
        self.rank = rank
        d_out = self.out_features // self.rank
        if self.out_features % self.rank != 0:
            raise ValueError(f"Output features ({self.out_features}) must be divisible by rank ({self.rank})")
        n_elements_out = (d_out * (d_out - 1)) // 2
        # Store only the upper triangular elements in a 1D tensor
        self.Q_left = nn.Parameter(torch.zeros(self.rank, n_elements_out,
                            device=linear_layer.weight.device,
                            dtype=linear_layer.weight.dtype))

        d_in = self.in_features // self.rank
        if self.in_features % self.rank != 0:
            raise ValueError(f"Input features ({self.in_features}) must be divisible by rank ({self.rank})")
        n_elements_in = (d_in * (d_in - 1)) // 2
        # Store only the upper triangular elements in a 1D tensor
        self.Q_right = nn.Parameter(torch.zeros(self.rank, n_elements_in,
                            device=linear_layer.weight.device,
                            dtype=linear_layer.weight.dtype))

        # op = matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device=device)
        if self.rank > 1:
            layout = torch.eye(self.rank).unsqueeze(0).to(torch.int64)
            self.op_left = matmul(layout, d_out, 'qoft_dsd', trans_a=False, trans_b=False, device=linear_layer.weight.device)
            self.op_right = matmul(layout, d_in, 'qoft_dds', trans_a=False, trans_b=False, device=linear_layer.weight.device)

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
        
        # Initialize indices (will be updated periodically)
        # self.register_buffer('selected_indices', torch.arange(self.rank, device=linear_layer.weight.device))
        self.steps_since_update = 0
        self.update_indices_gap = 50
    
    def update_indices(self):
        """Update the selected indices randomly."""
        with torch.no_grad():
            self.selected_indices = torch.randperm(self.in_features, device=self.weight.device)[:self.rank]
    
    
    def get_cayley_transform(self, mode='all'):
        """
        Get Cayley transform matrices based on specified mode.
        Args:
            mode (str): One of 'all', 'left', or 'right' to specify which transforms to compute
        Returns:
            Tuple of (R_left, R_right) where each R is a batch of smaller orthogonal matrices
            R_left shape: [d_out, rank, rank] where d_out = out_features // rank
            R_right shape: [d_in, rank, rank] where d_in = in_features // rank
        """
        R_left = None
        R_right = None
        
        # Process left transform if needed
        if mode in ['all', 'left']:
            d = self.out_features // self.rank
            Q_left = self.Q_left.to(torch.float32)  # Shape: [d, n_elements]
            
            # Create batch of skew-symmetric matrices
            Q_blocks = torch.zeros(self.rank, d, d, 
                                 device=self.Q_left.device, 
                                 dtype=torch.float32)
            triu_indices = torch.triu_indices(d, d, offset=1)
            Q_blocks[:, triu_indices[0], triu_indices[1]] = Q_left
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Cayley transform for all blocks at once
            I = torch.eye(d, device=self.Q_left.device, dtype=torch.float32)
            R_left = torch.linalg.solve(I - Q_blocks, I + Q_blocks).transpose(-2, -1)
            R_left = R_left.to(self.Q_left.dtype)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            d = self.in_features // self.rank
            Q_right = self.Q_right.to(torch.float32)  # Shape: [d, n_elements]
            
            # Create batch of skew-symmetric matrices
            Q_blocks = torch.zeros(self.rank, d, d,
                                 device=self.Q_right.device,
                                 dtype=torch.float32)
            triu_indices = torch.triu_indices(d, d, offset=1)
            Q_blocks[:, triu_indices[0], triu_indices[1]] = Q_right
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Cayley transform for all blocks at once
            I = torch.eye(d, device=self.Q_right.device, dtype=torch.float32)
            R_right = torch.linalg.solve(I - Q_blocks, I + Q_blocks).transpose(-2, -1)
            R_right = R_right.to(self.Q_right.dtype)

        return R_left, R_right
    

    def forward(self, x):
        '''
        # First use the current R to update layer_weight
        if hasattr(self, 'update_indices_gap') and self.steps_since_update % self.update_indices_gap == (self.update_indices_gap - 1):
            with torch.no_grad():
                R = self.get_cayley_transform()
                selected_weights = self.weight[:, self.selected_indices].to(R.dtype)  # Match dtype
                transformed_selected = torch.matmul(R, selected_weights.t()) # * self.s
                self.weight.data[:, self.selected_indices] = transformed_selected.t().to(self.weight.dtype) 

                self.update_indices()
                self.steps_since_update = 0
                self.Q.zero_()
                self.s.data.fill_(1.0)

            # breakpoint() # Match dtype when updating
        '''
        # Determine whether to use left or right transform based on step count
        # use_left = (self.steps_since_update // self.update_indices_gap) % 2 == 0 

        # Get orthogonal matrix R through Cayley transform
        # R_left, R_right = self.get_cayley_transform(mode='left' if use_left else 'right')
        R_left, R_right = self.get_cayley_transform()

        # Create a copy of the weight matrix on the correct device and dtype
        transformed_weight = self.weight.to(dtype=x.dtype, device=x.device)
        
        # Two-step matrix multiplication
        R_left_bs = torch.block_diag(*R_left)
        R_right_bs = torch.block_diag(*R_right)
        temp = transformed_weight @ R_right_bs
        transformed_weight = R_left_bs @ temp
        
        # if self.rank == 1:
        # temp = transformed_weight @ R_right.squeeze()
        # transformed_weight = R_left.squeeze() @ temp
        # else:
        #     transformed_weight = self.op_right(transformed_weight.unsqueeze(0), R_right)
        #     transformed_weight = self.op_left(R_left, transformed_weight)
        # Apply per-neuron scaling
        # Scale shape is [out_features, 1], which will broadcast across the in_features dimension
        # scaled_weight = transformed_weight * (1.0 + (self.scale - 1.0) * self.scale_scale)
        # scaled_weight = transformed_weight * (1.0 + self.scale) # * self.scale_scale)
        # scaled_weight = transformed_weight * self.scale

        # Apply R only to the selected subset of weights
        # selected_weights = self.weight[:, self.selected_indices].to(R.dtype)  # Match dtype
        # transformed_selected = torch.matmul(R, selected_weights.t()) * self.s
        # self.weight.data[:, self.selected_indices] = transformed_selected.t().to(self.weight.dtype) 

        self.steps_since_update += 1

        # Regular linear transformation with transformed weight
        return F.linear(x, transformed_weight.squeeze(), self.bias)


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
    
    data_dir = "/lustre/fast/fast/zqiu/GaLore/c4/en"
    import glob
    
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
            # Load a single file with parallel processing
            file_dataset = datasets.load_dataset(
                "json",
                data_files=file_path,
                features=features,
                streaming=False,
                cache_dir=None,
                keep_in_memory=True,
                num_proc=os.cpu_count()-1,  # Use multiple cores for loading
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
    parser.add_argument("--soft_rank", type=int, default=128)
    parser.add_argument("--update_indices_gap", type=int, default=5)
    parser.add_argument("--soft_scale", type=float, default=1.0)
    parser.add_argument("--soft_proj_type", type=str, default="std")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    # val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True) #DGX
    # val_data = val_data.shuffle(seed=42)
    val_data = load_local_data(split='validation', seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
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

    return total_loss, evaluated_on_tokens



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
        wandb.init(project="soft-c4", name=args.save_dir.split('/')[-1])
        
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


    if 'galore' in args.optimizer.lower() or args.optimizer.lower() == "adamw_baseline":
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
        param_groups = [
            # {'params': regular_params}, 
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
            new_layer = CayleyLinear(module, rank=args.soft_rank)
            # Set update_indices_gap
            new_layer.update_indices_gap = args.update_indices_gap
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
        param_groups = [
            # {'params': regular_params, 'lr': args.lr * 10},
            {'params': soft_params, 'lr': args.lr, 'soft_rank': args.soft_rank},  # Could use different lr for Q
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
    elif args.optimizer.lower() == "adamw_baseline":
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "galore_adamw" or args.optimizer.lower() == "only_galore" or args.optimizer.lower() == "no_galore":
        # redefine way to call galore_adamw
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "soft_adamw" or args.optimizer.lower() == "only_soft" or args.optimizer.lower() == "no_soft":
        # redefine way to call soft_adamw
        # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = SOFTAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = SOFTCayleyAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = SOFTCayleyAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
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
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

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

        # add grad clipping
        # if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        '''
        # The below code is only executed during the update step
        if global_rank == 0:
            metrics = {}
            
            # Collect statistics for all CayleyLinear layers
            for i, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, CayleyLinear):
                    with torch.no_grad():
                        layer_name = f"layer_{i}"
                        
                        # Scale statistics
                        metrics[f'scale_norm/{layer_name}'] = torch.norm(module.scale).item()
                        metrics[f'scale_mean/{layer_name}'] = torch.mean(module.scale).item()
                        
                        # Scale gradient statistics if available
                        if module.scale.grad is not None:
                            metrics[f'scale_grad_norm/{layer_name}'] = torch.norm(module.scale.grad).item()
                            metrics[f'scale_grad_mean/{layer_name}'] = torch.mean(torch.abs(module.scale.grad)).item()

            # Add update step
            metrics['update_step'] = update_step
            
            # Log to wandb
            wandb.log(metrics, step=global_step)


        # Gradient inspection
        if global_rank == 0:
            metrics = {}
            
            # Collect statistics for all Q parameters and regular weights
            for i, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, CayleyLinear):
                    if module.Q.grad is not None:
                        layer_name = f"layer_{i}"
                        
                        # Gradient statistics
                        metrics[f'Q_grad_norm/{layer_name}'] = torch.norm(module.Q.grad).item()
                        metrics[f'Q_grad_mean/{layer_name}'] = torch.mean(torch.abs(module.Q.grad)).item()
                        
                        # Check orthogonality
                        with torch.no_grad():
                            R = module.get_cayley_transform()
                            I = torch.eye(module.rank, device=R.device, dtype=R.dtype)
                            metrics[f'R_orthogonality_error/{layer_name}'] = torch.norm(R.T @ R - I).item()
                
                elif isinstance(module, nn.Linear):
                    if module.weight.grad is not None:
                        layer_name = f"layer_{i}"
                        
                        # Weight gradient statistics
                        metrics[f'W_grad_norm/{layer_name}'] = torch.norm(module.weight.grad).item()
                        metrics[f'W_grad_mean/{layer_name}'] = torch.mean(torch.abs(module.weight.grad)).item()
                        
                        # Weight norm
                        with torch.no_grad():
                            metrics[f'W_norm/{layer_name}'] = torch.norm(module.weight).item()

            # Add update step
            metrics['update_step'] = update_step
            
            # Log to wandb
            wandb.log(metrics, step=global_step)

        # Calculate and log spectral norms
        if global_rank == 0:
            metrics = {}
            
            # Collect gradient spectral norms for all linear layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, CayleyLinear)):
                    if hasattr(module, 'weight') and module.weight.grad is not None:
                        # Calculate spectral norm using torch.linalg.svdvals
                        with torch.no_grad():
                            grad = module.weight.grad
                            if isinstance(module, CayleyLinear):
                                # For CayleyLinear, we need to handle the gradient differently
                                transformed_grad = grad.clone()
                                R = module.get_cayley_transform()
                                transformed_grad[:, module.selected_indices] = transformed_grad[:, module.selected_indices] @ R
                                grad = transformed_grad * (1.0 + module.scale)
                            
                            # Cast to float32 for SVD computation
                            grad_float32 = grad.to(torch.float32)
                            
                            # Calculate spectral norm (largest singular value)
                            s = torch.linalg.svdvals(grad_float32)
                            spectral_norm = s[0].item()
                            
                            # Log spectral norm
                            metrics[f'grad_spectral_norm/{name}'] = spectral_norm
                            
                            # Also log Frobenius norm for comparison
                            frob_norm = torch.norm(grad_float32).item()
                            metrics[f'grad_frobenius_norm/{name}'] = frob_norm
                            
                            # Log ratio of spectral to Frobenius norm
                            metrics[f'grad_norm_ratio/{name}'] = spectral_norm / frob_norm if frob_norm > 0 else 0

            # Add these metrics to the existing wandb log
            wandb.log(metrics, step=global_step)
        '''

        if global_rank == 0: pbar.update(1)
        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            # Check if model is wrapped in DDP
            if hasattr(model, 'module'):
                model.module.save_pretrained(current_model_directory)
            else:
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
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

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

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        # Check if model is wrapped in DDP
        if hasattr(model, 'module'):
            model.module.save_pretrained(current_model_directory)
        else:
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

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)

