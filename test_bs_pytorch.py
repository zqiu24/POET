import torch
import time
import torch.nn as nn

def test_block_transform_methods():
    """
    Test function to compare the optimized block transform method against
    the torch.block_diag implementation, verifying correctness and measuring performance.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 32
    in_features = 512
    out_features = 384
    rank = 4
    n_runs = 10  # Number of runs for timing
    
    # Calculate block dimensions
    d_out = out_features // rank
    d_in = in_features // rank
    
    # Create random weight matrix
    weight = torch.randn(out_features, in_features)
    
    # Create random transform blocks
    R_left = torch.randn(rank, d_out, d_out)
    R_right = torch.randn(rank, d_in, d_in)
    
    # Define the block_diag implementation
    def transform_with_block_diag(weight, R_left, R_right):
        # Create full block diagonal matrices
        R_left_full = torch.block_diag(*[R_left[i] for i in range(rank)])
        R_right_full = torch.block_diag(*[R_right[i] for i in range(rank)])
        
        # Apply transforms
        temp = torch.matmul(weight, R_right_full)
        result = torch.matmul(R_left_full, temp)
        
        return result
    
    # Define the optimized implementation
    def transform_optimized(weight, R_left, R_right):
        # Create output tensor
        transformed_weight = weight.clone()
        
        # Process only diagonal blocks
        for i in range(rank):
            # Extract block
            weight_block = weight[i*d_out:(i+1)*d_out, i*d_in:(i+1)*d_in]
            
            # Apply transforms
            temp = torch.matmul(weight_block, R_right[i])
            transformed_block = torch.matmul(R_left[i], temp)
            
            # Place back
            transformed_weight[i*d_out:(i+1)*d_out, i*d_in:(i+1)*d_in] = transformed_block
        
        return transformed_weight
    
    # Verify correctness
    print("Verifying correctness...")
    result_block_diag = transform_with_block_diag(weight, R_left, R_right)
    result_optimized = transform_optimized(weight, R_left, R_right)
    
    # Check if results match
    diff = torch.abs(result_block_diag - result_optimized).max().item()
    print(f"Maximum absolute difference: {diff:.10f}")
    if diff < 1e-5:
        print("✓ Results match within tolerance")
    else:
        print("✗ Results do not match!")
    
    # Measure performance
    print("\nMeasuring performance:")
    
    # Time block_diag implementation
    start_time = time.time()
    for _ in range(n_runs):
        _ = transform_with_block_diag(weight, R_left, R_right)
    block_diag_time = (time.time() - start_time) / n_runs
    print(f"Average time using block_diag: {block_diag_time*1000:.2f} ms")
    
    # Time optimized implementation
    start_time = time.time()
    for _ in range(n_runs):
        _ = transform_optimized(weight, R_left, R_right)
    optimized_time = (time.time() - start_time) / n_runs
    print(f"Average time using optimized: {optimized_time*1000:.2f} ms")
    
    # Calculate speedup
    speedup = block_diag_time / optimized_time
    print(f"Speedup: {speedup:.2f}x faster")
    
    # Memory usage comparison
    print("\nEstimated memory comparison:")
    block_diag_memory = out_features * out_features + in_features * in_features + out_features * in_features
    optimized_memory = rank * (d_out * d_out + d_in * d_in) + out_features * in_features
    memory_reduction = 1 - (optimized_memory / block_diag_memory)
    print(f"Block diag memory usage: {block_diag_memory} elements")
    print(f"Optimized memory usage: {optimized_memory} elements")
    print(f"Memory reduction: {memory_reduction*100:.2f}%")

if __name__ == "__main__":
    test_block_transform_methods()