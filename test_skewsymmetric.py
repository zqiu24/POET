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
    upper_val = tl.load(vec_ptr + upper_idx, mask=upper_mask & full_mask, other=0.0)
    
    # Lower triangle
    lower_mask = i > j
    lower_idx = j * (2 * N - j - 1) // 2 + (i - j - 1)
    lower_val = -tl.load(vec_ptr + lower_idx, mask=lower_mask & full_mask, other=0.0)
    
    result = tl.where(upper_mask, upper_val, tl.where(lower_mask, lower_val, 0.0))
    mat_ptrs = mat_ptr + i * stride_row + j * stride_col
    tl.store(mat_ptrs, result, mask=full_mask)

# --------------------------
# Triton Backward Kernel
# --------------------------


@triton.jit
def skew_symmetric_backward_kernel(
    grad_mat_ptr,
    grad_vec_ptr,
    N: tl.int32,
    stride_row: tl.int32,
    stride_col: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = N * (N - 1) // 2
    mask = offs < total_elements
    
    k = offs
    # Correct type conversion using .to()
    N_float = N.to(tl.float32)  # Fixed here
    k_float = k.to(tl.float32)
    
    # Correct quadratic formula implementation
    a = 2.0 * N_float - 1.0
    sqrt_val = tl.sqrt(a * a - 8.0 * k_float)
    i = (a - sqrt_val) / 2.0
    i = tl.floor(i).to(tl.int32)
    
    # Calculate j using remaining offset
    triangular_num = i * (2 * N - i - 1) // 2
    j = k - triangular_num + i + 1
    
    # Boundary validation
    valid = (i >= 0) & (j < N) & (i < j)
    mask = mask & valid
    
    # Gradient calculation
    grad_upper = tl.load(
        grad_mat_ptr + i * stride_row + j * stride_col,
        mask=mask,
        other=0.0
    )
    grad_lower = tl.load(
        grad_mat_ptr + j * stride_row + i * stride_col,
        mask=mask,
        other=0.0
    )
    tl.store(grad_vec_ptr + k, grad_upper - grad_lower, mask=mask)



@triton.jit
def skew_symmetric_backward_kernel_correct_but_slow(
    grad_mat_ptr,
    grad_vec_ptr,
    N,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = N * (N - 1) // 2
    mask = offs < total_elements
    
    # Reconstruct i,j from vector index using a simpler approach
    # We'll iterate through the upper triangular elements row by row
    k = offs
    
    # First, determine which row this element belongs to
    # This is a simpler formula that should compile in Triton
    i = tl.zeros_like(k)
    j = tl.zeros_like(k)
    
    # For each element, compute its position
    for row in range(N-1):
        row_start = row * (2*N - row - 1) // 2
        row_size = N - row - 1
        row_mask = (k >= row_start) & (k < row_start + row_size)
        
        # If in this row, calculate column
        i = tl.where(row_mask, row, i)
        j = tl.where(row_mask, row + 1 + (k - row_start), j)
    
    # Gradient calculation
    grad_upper = tl.load(grad_mat_ptr + i * stride_row + j * stride_col, mask=mask, other=0.0)
    grad_lower = tl.load(grad_mat_ptr + j * stride_row + i * stride_col, mask=mask, other=0.0)
    
    # The gradient is (upper - lower) because of the skew-symmetric property
    tl.store(grad_vec_ptr + offs, grad_upper - grad_lower, mask=mask)


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
            vec_ptr=vec,
            mat_ptr=mat,
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
            grad_mat_ptr=grad_output.contiguous(),  # Ensure contiguous
            grad_vec_ptr=grad_vec,
            N=N,
            stride_row=grad_output.stride(0),
            stride_col=grad_output.stride(1),
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
    skew_mat = mat - mat.T
    return skew_mat

# --------------------------
# Testing & Benchmarking
# --------------------------
def test_correctness(N=512):
    vec = torch.randn(N*(N-1)//2, device='cuda', dtype=torch.float32, requires_grad=True)  # FIX HERE
    
    # Forward
    mat_triton = SkewSymmetric.apply(vec, N)
    mat_pytorch = pytorch_skew(vec, N)
    
    # Backward
    grad_output = torch.randn_like(mat_triton)
    
    # Triton backward
    vec.grad = None
    mat_triton.backward(grad_output)
    grad_triton = vec.grad.clone()
    
    # PyTorch backward
    vec.grad = None
    mat_pytorch.backward(grad_output)
    grad_pytorch = vec.grad.clone()
    
    # Verification
    assert torch.allclose(mat_triton, mat_pytorch), "Forward mismatch"
    print('mat_triton', grad_triton)
    print('mat_pytorch', grad_pytorch)
    assert torch.allclose(grad_triton, grad_pytorch, atol=1e-5), f"Backward mismatch"

def benchmark_forward(N=8192):
    vec = torch.randn(N*(N-1)//2, device='cuda')
    
    # Warmup
    for _ in range(3):
        _ = SkewSymmetric.apply(vec, N)
        _ = pytorch_skew(vec, N)
    torch.cuda.synchronize()
    
    # Triton
    start = time.time()
    mat = SkewSymmetric.apply(vec, N)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # PyTorch
    start = time.time()
    mat_pytorch = pytorch_skew(vec, N)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    print(f"Forward N={N}")
    print(f"Triton: {triton_time*1000:.2f}ms")
    print(f"PyTorch: {pytorch_time*1000:.2f}ms")
    print(f"Speedup: {pytorch_time/triton_time:.1f}x\n")

def benchmark_backward(N=8192):
    vec = torch.randn(N*(N-1)//2, device='cuda', requires_grad=True)
    grad_output = torch.randn((N, N), device='cuda')
    
    # Triton
    # Warmup
    for _ in range(3):
        mat_triton = SkewSymmetric.apply(vec.clone().requires_grad_(), N)
        mat_triton.backward(grad_output)
    torch.cuda.synchronize()
    
    # Benchmark
    vec_triton = vec.clone().requires_grad_()
    start = time.time()
    mat_triton = SkewSymmetric.apply(vec_triton, N)
    mat_triton.backward(grad_output)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # PyTorch
    # Warmup
    for _ in range(3):
        mat_pytorch = pytorch_skew(vec.clone().requires_grad_(), N)
        mat_pytorch.backward(grad_output)
    torch.cuda.synchronize()
    
    # Benchmark
    vec_pytorch = vec.clone().requires_grad_()
    start = time.time()
    mat_pytorch = pytorch_skew(vec_pytorch, N)
    mat_pytorch.backward(grad_output)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    print(f"Backward N={N}")
    print(f"Triton: {triton_time*1000:.2f}ms")
    print(f"PyTorch: {pytorch_time*1000:.2f}ms")
    print(f"Speedup: {pytorch_time/triton_time:.1f}x\n")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("Running correctness test...")
    test_correctness()
    
    # print("Benchmarking large matrix:")
    benchmark_forward(8192)
    benchmark_backward(8192)