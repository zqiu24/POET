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
from soft_torch import matmul

transformers.logging.set_verbosity_error()


class CayleyLinear(nn.Linear):
    # Class-level variable to track steps across all instances
    global_step_counter = 0
    
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
        if self.rank > 10000:
            layout = torch.eye(self.rank).unsqueeze(0).to(torch.int64)
            self.op_left = matmul(layout, d_out, 'qoft_dsd', trans_a=False, trans_b=False, device=linear_layer.weight.device)
            self.op_right = matmul(layout, d_in, 'qoft_dds', trans_a=False, trans_b=False, device=linear_layer.weight.device)

        # Pre-compute and store triu_indices as buffers
        i_out, j_out = torch.triu_indices(d_out, d_out, offset=1)
        self.triu_indices_out_i = i_out
        self.triu_indices_out_j = j_out
        
        i_in, j_in = torch.triu_indices(d_in, d_in, offset=1)
        self.triu_indices_in_i = i_in
        self.triu_indices_in_j = j_in
        
        self.use_neumann = False
        self.num_neumann_terms = 1
        self.reset_R = False
        self.update_reset_R_gap = 0
        self.selected_indices = torch.arange(self.rank, device=linear_layer.weight.device)

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
        
        # Initialize indices (will be updated periodically)
        # self.register_buffer('selected_indices', torch.arange(self.rank, device=linear_layer.weight.device))
    
    # def update_indices(self):
    #     """Update the selected indices randomly."""
    #     with torch.no_grad():
    #         self.selected_indices = torch.randperm(self.in_features, device=self.weight.device)[:self.rank]

    def get_cayley_transform_neumann(self, mode='all'):
        """
        Get Cayley transform matrices based on specified mode using Neumann series.
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
            
            # More efficient skew-symmetric matrix construction
            Q_blocks = torch.zeros(self.rank, d, d, device=self.Q_left.device, dtype=self.Q_left.dtype)
            # Fill only upper triangular part
            Q_blocks[:, self.triu_indices_out_i, self.triu_indices_out_j] = self.Q_left
            # Create the skew-symmetric matrix in one operation (A - A^T)
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Cayley transform using Neumann series approximation
            I = torch.eye(d, device=self.Q_left.device, dtype=self.Q_left.dtype)
            
            # Initialize the result with identity
            R_left = I.clone().expand(self.rank, d, d)
            
            # Compute the series expansion up to the specified number of terms
            Q_power = Q_blocks.clone()  # Start with Q^1
            coeff = 2.0  # Coefficient for the first term (Q^1)
            
            for i in range(1, self.num_neumann_terms):
                R_left = R_left + coeff * Q_power
                
                if i < self.num_neumann_terms - 1:  # Prepare next term if needed
                    Q_power = torch.matmul(Q_power, Q_blocks)  # Compute next power Q^(i+1)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            d = self.in_features // self.rank
            
            # More efficient skew-symmetric matrix construction
            Q_blocks = torch.zeros(self.rank, d, d, device=self.Q_right.device, dtype=self.Q_right.dtype)
            # Fill only upper triangular part
            Q_blocks[:, self.triu_indices_in_i, self.triu_indices_in_j] = self.Q_right
            # Create the skew-symmetric matrix in one operation (A - A^T)
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
        
            # Compute Cayley transform using Neumann series approximation
            I = torch.eye(d, device=self.Q_right.device, dtype=self.Q_right.dtype)
            
            # Initialize the result with identity
            R_right = I.clone().expand(self.rank, d, d)
            
            # Compute the series expansion up to the specified number of terms
            Q_power = Q_blocks.clone()  # Start with Q^1
            coeff = 2.0  # Coefficient for the first term (Q^1)
            
            for i in range(1, self.num_neumann_terms):
                R_right = R_right + coeff * Q_power
                
                if i < self.num_neumann_terms - 1:  # Prepare next term if needed
                    Q_power = torch.matmul(Q_power, Q_blocks)  # Compute next power Q^(i+1)

        return R_left, R_right
    
    
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
            
            # More efficient skew-symmetric matrix construction
            Q_blocks = torch.zeros(self.rank, d, d, device=self.Q_left.device, dtype=torch.float32)
            # Fill only upper triangular part
            Q_blocks[:, self.triu_indices_out_i, self.triu_indices_out_j] = Q_left
            # Create the skew-symmetric matrix in one operation (A - A^T)
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Cayley transform for all blocks at once
            I = torch.eye(d, device=self.Q_left.device, dtype=torch.float32)
            R_left = torch.linalg.solve(I - Q_blocks, I + Q_blocks).transpose(-2, -1)
            R_left = R_left.to(self.Q_left.dtype)
        
        # Process right transform if needed
        if mode in ['all', 'right']:
            d = self.in_features // self.rank
            Q_right = self.Q_right.to(torch.float32)  # Shape: [d, n_elements]
            
            # More efficient skew-symmetric matrix construction
            Q_blocks = torch.zeros(self.rank, d, d, device=self.Q_right.device, dtype=torch.float32)
            # Fill only upper triangular part
            Q_blocks[:, self.triu_indices_in_i, self.triu_indices_in_j] = Q_right
            # Create the skew-symmetric matrix in one operation (A - A^T)
            Q_blocks = Q_blocks - Q_blocks.transpose(-2, -1)
            
            # Compute Cayley transform for all blocks at once
            I = torch.eye(d, device=self.Q_right.device, dtype=torch.float32)
            R_right = torch.linalg.solve(I - Q_blocks, I + Q_blocks).transpose(-2, -1)
            R_right = R_right.to(self.Q_right.dtype)

        return R_left, R_right


    def forward(self, x):
        # Access the class-level counter
        global_step = CayleyLinear.global_step_counter
        
        # If reset_R flag is set, update weight directly and reset parameters
        if self.reset_R and self.update_reset_R_gap > 0 and global_step % self.update_reset_R_gap == 0 and global_step > 0:
            print(f"Resetting R at global step {global_step}")
            with torch.no_grad():
                # Compute transforms based on current parameters
                if self.use_neumann:
                    R_left, R_right = self.get_cayley_transform_neumann()
                else:
                    R_left, R_right = self.get_cayley_transform()

                # Apply transforms to weight directly
                transformed_weight = self.weight.clone()
                
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

        # Get transforms - will use cached values if parameters haven't changed
        if self.use_neumann:
            R_left, R_right = self.get_cayley_transform_neumann()
        else:
            R_left, R_right = self.get_cayley_transform()
        
        # Create block diagonal matrices
        R_left_bs = torch.block_diag(*R_left)
        R_right_bs = torch.block_diag(*R_right)
        
        # Apply transforms to weight
        transformed_weight = self.weight.to(dtype=x.dtype, device=x.device)
        temp = torch.matmul(transformed_weight, R_right_bs)
        transformed_weight = torch.matmul(R_left_bs, temp)

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
        cache_file_name=None,  # Disable caching during preprocessing
        load_from_cache_file=False  # Don't load from cache
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
            new_layer = CayleyLinear(module, rank=args.soft_rank)
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
                {'params': soft_params, 'lr': args.soft_lr, 'soft_rank': args.soft_rank, 'scale': args.soft_scale},
            ]
        else:
            param_groups = [
                {'params': soft_params, 'lr': args.soft_lr, 'soft_rank': args.soft_rank, 'scale': args.soft_scale},
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
    elif args.optimizer.lower() == "soft_adamw_neumann" or args.optimizer.lower() == "only_soft_adamw_neumann":
        # redefine way to call soft_adamw
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = SOFTAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
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

        '''
        # Gradient inspection
        if global_rank == 0:
            metrics = {}
            
            # Collect statistics for all Q parameters and regular weights
            for i, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, CayleyLinear):
                    if module.Q_left.grad is not None and module.Q_right.grad is not None:
                        layer_name = f"layer_{i}"
                        
                        # Gradient statistics
                        metrics[f'Q_left_grad_norm/{layer_name}'] = torch.norm(module.Q_left.grad).item()
                        metrics[f'Q_left_grad_mean/{layer_name}'] = torch.mean(torch.abs(module.Q_left.grad)).item()
                        metrics[f'Q_right_grad_norm/{layer_name}'] = torch.norm(module.Q_right.grad).item()
                        metrics[f'Q_right_grad_mean/{layer_name}'] = torch.mean(torch.abs(module.Q_right.grad)).item()
                        
                        # Check orthogonality
                        with torch.no_grad():
                            R_left, R_right = module.get_cayley_transform_neumann()
                            R_left_bs = torch.block_diag(*R_left)
                            R_right_bs = torch.block_diag(*R_right)

                            if len(R_left_bs.shape) != 2:
                                breakpoint()
                            if len(R_right_bs.shape) != 2:
                                breakpoint()

                            if R_left_bs.shape[0] != R_left_bs.shape[1]:
                                breakpoint()
                            if R_right_bs.shape[0] != R_right_bs.shape[1]:
                                breakpoint()

                            I = torch.eye(R_left_bs.shape[0], device=R_left_bs.device, dtype=R_left_bs.dtype)
                            metrics[f'R_left_orthogonality_error/{layer_name}'] = torch.norm(R_left_bs.T @ R_left_bs - I).item()
                            
                            I = torch.eye(R_right_bs.shape[0], device=R_right_bs.device, dtype=R_right_bs.dtype)
                            metrics[f'R_right_orthogonality_error/{layer_name}'] = torch.norm(R_right_bs.T @ R_right_bs - I).item()

            # Add update step
            metrics['update_step'] = update_step
            
            # Log to wandb
            wandb.log(metrics, step=global_step)
        '''

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

