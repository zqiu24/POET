import torch
import triton
import triton.language as tl
from torch.autograd import Function
import time

# --------------------------
# Triton Forward Kernel
# --------------------------
@triton.jit
def vector_to_skew_symmetric_kernel(
    vec_ptr,
    mat_ptr,
    N,
    stride_vec,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_m = offs_m < N
    mask_n = offs_n < N
    full_mask = mask_m[:, None] & mask_n[None, :]
    
    i = offs_m[:, None]
    j = offs_n[None, :]
    
    # Upper triangle
    upper_mask = i < j
    upper_idx = i * (2 * N - i - 1) // 2 + (j - i - 1)
    upper_val = tl.load(vec_ptr + upper_idx * stride_vec, mask=upper_mask & full_mask, other=0.0)
    
    # Lower triangle
    lower_mask = i > j
    lower_idx = j * (2 * N - j - 1) // 2 + (i - j - 1)
    lower_val = -tl.load(vec_ptr + lower_idx * stride_vec, mask=lower_mask & full_mask, other=0.0)
    
    result = tl.where(upper_mask, upper_val, tl.where(lower_mask, lower_val, 0.0))
    mat_ptrs = mat_ptr + i * stride_row + j * stride_col
    tl.store(mat_ptrs, result, mask=full_mask)

# --------------------------
# Triton Backward Kernel (Fixed)
# --------------------------
@triton.jit
def skew_symmetric_backward_kernel(
    grad_mat_ptr,
    grad_vec_ptr,
    N,
    stride_row,
    stride_col,
    stride_vec,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = N * (N - 1) // 2
    mask = offs < total_elements
    
    # Reconstruct i,j from vector index
    k = offs
    
    # Fixed formula: Compute row i directly
    # For a given k, find i such that row i contains k
    i = ((2*N - 1) - tl.sqrt((2*N - 1)**2 - 8*k)) / 2
    i = tl.floor(i + 1e-5).to(tl.int32)  # Add a small epsilon to handle numerical precision issues
    
    # Calculate the starting index for row i
    start_i = i * (2 * N - i - 1) // 2
    
    # Calculate j based on the offset from start_i
    j = k - start_i + i + 1
    
    # Gradient calculation
    grad_upper = tl.load(grad_mat_ptr + i * stride_row + j * stride_col, mask=mask)
    grad_lower = tl.load(grad_mat_ptr + j * stride_row + i * stride_col, mask=mask)
    tl.store(grad_vec_ptr + k * stride_vec, grad_upper - grad_lower, mask=mask)

# --------------------------
# Autograd Function (fixed)
# --------------------------
class SkewSymmetric(Function):
    @staticmethod
    def forward(ctx, vec, N):
        ctx.N = N
        mat = torch.zeros((N, N), device=vec.device, dtype=vec.dtype)
        
        # Launch kernel
        BLOCK_SIZE = 32
        grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
        vector_to_skew_symmetric_kernel[grid](
            vec_ptr=vec.data_ptr(),
            mat_ptr=mat.data_ptr(),
            N=N,
            stride_vec=vec.stride(0),
            stride_row=mat.stride(0),
            stride_col=mat.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(vec)
        return mat

    @staticmethod
    def backward(ctx, grad_output):
        vec, = ctx.saved_tensors
        N = ctx.N
        grad_vec = torch.zeros_like(vec)
        
        # Launch backward kernel
        BLOCK_SIZE = 128
        total_elements = N * (N - 1) // 2
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        skew_symmetric_backward_kernel[grid](
            grad_mat_ptr=grad_output.contiguous().data_ptr(),  # Ensure contiguous
            grad_vec_ptr=grad_vec.data_ptr(),
            N=N,
            stride_row=grad_output.stride(0),
            stride_col=grad_output.stride(1),
            stride_vec=grad_vec.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return grad_vec, None

# --------------------------
# PyTorch Baseline
# --------------------------
def pytorch_skew(vec, N):
    mat = torch.zeros((N, N), device=vec.device, dtype=vec.dtype)
    rows, cols = torch.triu_indices(N, N, 1, device=vec.device)
    mat[rows, cols] = vec
    mat[cols, rows] = -vec
    return mat

# --------------------------
# Convenience function
# --------------------------
def skew_symmetric(vec, N):
    """
    Convert a vector of upper triangular elements to a skew-symmetric matrix.
    
    Args:
        vec: Vector of upper triangular elements (N*(N-1)/2 elements)
        N: Size of the output matrix (N x N)
        
    Returns:
        A skew-symmetric matrix of size N x N
    """
    return SkewSymmetric.apply(vec, N)


def test_correctness(N=512):
    """Test the correctness of the Triton implementation against PyTorch"""
    print(f"Testing correctness with N={N}...")
    
    # Create random vector for testing
    vec = torch.randn(N*(N-1)//2, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Forward pass
    mat_triton = skew_symmetric(vec, N)
    mat_pytorch = pytorch_skew(vec, N)
    
    # Check if forward results match
    forward_diff = torch.abs(mat_triton - mat_pytorch).max().item()
    print(f"Forward max difference: {forward_diff:.8f}")
    assert torch.allclose(mat_triton, mat_pytorch, atol=1e-5), "Forward pass mismatch"
    
    # Random gradient for backward pass
    grad_output = torch.randn_like(mat_triton)
    
    # Triton backward
    vec.grad = None
    mat_triton.backward(grad_output)
    grad_triton = vec.grad.clone()
    
    # PyTorch backward
    vec.grad = None
    mat_pytorch.backward(grad_output)
    grad_pytorch = vec.grad.clone()
    
    # Check if backward results match
    backward_diff = torch.abs(grad_triton - grad_pytorch).max().item()
    print(f"Backward max difference: {backward_diff:.8f}")
    
    # Print first few elements of both gradients
    print("First 5 elements of Triton gradient:", grad_triton[:5])
    print("First 5 elements of PyTorch gradient:", grad_pytorch[:5])
    
    # Print last few elements of both gradients
    print("Last 5 elements of Triton gradient:", grad_triton[-5:])
    print("Last 5 elements of PyTorch gradient:", grad_pytorch[-5:])
    
    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-5), "Backward pass mismatch"
    print("✓ All tests passed!")
    
def benchmark(N=8192):
    """Benchmark the Triton implementation against PyTorch"""
    print(f"\nBenchmarking with N={N}...")
    
    vec = torch.randn(N*(N-1)//2, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Warmup
    for _ in range(3):
        _ = skew_symmetric(vec, N)
        _ = pytorch_skew(vec, N)
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    import time
    
    # Triton forward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        mat_triton = skew_symmetric(vec, N)
    torch.cuda.synchronize()
    triton_forward_time = (time.time() - start) / 10
    
    # PyTorch forward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        mat_pytorch = pytorch_skew(vec, N)
    torch.cuda.synchronize()
    pytorch_forward_time = (time.time() - start) / 10
    
    print(f"Forward pass benchmark:")
    print(f"  Triton:  {triton_forward_time*1000:.2f} ms")
    print(f"  PyTorch: {pytorch_forward_time*1000:.2f} ms")
    print(f"  Speedup: {pytorch_forward_time/triton_forward_time:.2f}x")
    
    # Create matrices for backward pass
    mat_triton = skew_symmetric(vec, N)
    mat_pytorch = pytorch_skew(vec, N)
    grad_output = torch.randn_like(mat_triton)
    
    # Warmup backward
    for _ in range(3):
        vec.grad = None
        mat_triton.backward(grad_output)
        vec.grad = None
        mat_pytorch.backward(grad_output)
    torch.cuda.synchronize()
    
    # Triton backward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        vec.grad = None
        mat_triton.backward(grad_output)
    torch.cuda.synchronize()
    triton_backward_time = (time.time() - start) / 10
    
    # PyTorch backward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        vec.grad = None
        mat_pytorch.backward(grad_output)
    torch.cuda.synchronize()
    pytorch_backward_time = (time.time() - start) / 10
    
    print(f"Backward pass benchmark:")
    print(f"  Triton:  {triton_backward_time*1000:.2f} ms")
    print(f"  PyTorch: {pytorch_backward_time*1000:.2f} ms")
    print(f"  Speedup: {pytorch_backward_time/triton_backward_time:.2f}x")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test small matrix for correctness
    test_correctness(N=64)
    
    # Test medium matrix for correctness
    test_correctness(N=512)
    
    # Benchmark large matrix
    benchmark(N=8192)