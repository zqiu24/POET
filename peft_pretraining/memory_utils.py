import torch


def calculate_memory_usage(model, optimizer_name, dtype, trainable_params=None, param_groups=None, rank=None):
    """
    Calculate theoretical memory usage for parameters and optimizer states.
    
    Args:
        model: The model to calculate memory for
        optimizer_name: Name of the optimizer
        dtype: Data type of model parameters
        trainable_params: List of parameters passed to regular optimizers
        param_groups: List of parameter groups for optimizers like GaLore
        rank: Rank parameter for GaLore
    """
    param_size = 0
    param_count = 0
    optimizer_size = 0
    galore_size = 0  # Separate GaLore projection matrices memory
    soft_size = 0  # Separate SOFT projection matrices memory
    
    # Get bytes per parameter based on dtype
    bytes_per_param = 2 if dtype in [torch.bfloat16, torch.float16] else 4
    
    # Count both parameters and buffers
    for tensor in list(model.parameters()) + list(model.buffers()):
        param_count += tensor.numel()
        param_size += tensor.numel() * bytes_per_param
    
    # 2. Optimizer State Memory
    if param_groups is not None:
        for group in param_groups:
            params_in_group = group['params']   
            
            if 'rank' in group:  # GaLore parameters
                for param in params_in_group:
                    m, n = param.shape  # grad shape is (m x n)
                    if m >= n:
                        # Right projection: grad (m x n) @ ortho (n x rank) = (m x rank)
                        if '8bit' in optimizer_name.lower():
                            optimizer_size += 2 * (m * rank)  # exp_avg and exp_avg_sq in 8-bit
                        else:
                            optimizer_size += 2 * (m * rank) * bytes_per_param  # exp_avg and exp_avg_sq match dtype
                        galore_size += (n * rank) * bytes_per_param  # Projection matrix matches dtype
                    else:
                        # Left projection: ortho (rank x m) @ grad (m x n) = (rank x n)
                        if '8bit' in optimizer_name.lower():
                            optimizer_size += 2 * (rank * n)  # exp_avg and exp_avg_sq in 8-bit
                        else:
                            optimizer_size += 2 * (rank * n) * bytes_per_param  # exp_avg and exp_avg_sq match dtype
                        galore_size += (m * rank) * bytes_per_param  # Projection matrix matches dtype
                    galore_size += param.numel() * bytes_per_param  # full rank grad
            elif 'soft_rank' in group:
                for param in params_in_group:
                    # m, _ = param.shape  # grad shape is (m x n)
                    # soft_size += m * (m - 1) / 2 * 3 * bytes_per_param  # Projection matrix matches dtype
                    soft_size += param.numel() * bytes_per_param * 2
            else:  # Regular parameters
                for param in params_in_group:
                    if '8bit' in optimizer_name.lower():
                        optimizer_size += param.numel() * 1 * 2  # 8-bit states
                    else:
                        # Optimizer states match parameter dtype
                        optimizer_size += param.numel() * bytes_per_param * 2
    
    elif trainable_params is not None:
        for param in trainable_params:
            if '8bit' in optimizer_name.lower():
                optimizer_size += param.numel() * 1 * 2
            else:
                optimizer_size += param.numel() * bytes_per_param * 2
    
    total_size = param_size + optimizer_size + galore_size + soft_size
    
    return {
        'parameter_size_gb': param_size / (1024**3),
        'optimizer_size_gb': optimizer_size / (1024**3),
        'galore_size_gb': galore_size / (1024**3),
        'soft_size_gb': soft_size / (1024**3),
        'total_size_gb': total_size / (1024**3),
        'param_count_m': param_count / 1e6
    }