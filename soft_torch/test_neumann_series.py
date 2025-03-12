import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def create_skew_symmetric(tensor):
    """Create a skew-symmetric matrix from a tensor"""
    return 0.5 * (tensor - tensor.transpose(-2, -1))

def sync_cuda():
    """Synchronize CUDA for accurate timing"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def cayley_approach1(Q_blocks, num_terms):
    batch_size, n, n = Q_blocks.shape
    device, dtype = Q_blocks.device, Q_blocks.dtype
    R = torch.eye(n, device=device, dtype=dtype).expand(batch_size, n, n).clone()
    
    if num_terms > 1:
        R.add_(Q_blocks, alpha=2.0)
        
        if num_terms > 2:
            Q_squared = torch.bmm(Q_blocks, Q_blocks)
            R.add_(Q_squared, alpha=2.0)
            
            Q_power = Q_squared
            for i in range(3, num_terms):
                Q_power = torch.bmm(Q_power, Q_blocks)
                R.add_(Q_power, alpha=2.0)
    
    return R

def cayley_approach2(Q, num_terms):
    """
    Compute Cayley transform using simplified form
    Approach 2: I - 2Q + 2Q^2 - 2Q^3 + ...
    """
    batch_size, n, n = Q.shape
    device, dtype = Q.device, Q.dtype
    
    # Start with identity
    result = torch.eye(n, device=device, dtype=dtype).expand(batch_size, n, n).clone()
    
    if num_terms > 1:
        # First term: -2Q
        result.add_(Q, alpha=-2.0)
        
        if num_terms > 2:
            # Compute higher order terms
            Q_squared = torch.bmm(Q, Q)
            sign = 2.0  # Start with positive for Q^2
            
            # Add Q^2
            result.add_(Q_squared, alpha=sign)
            
            # Previous power
            Q_power = Q_squared
            
            # Remaining terms
            for k in range(3, num_terms):
                sign *= -1.0  # Alternate signs
                Q_power = torch.bmm(Q_power, Q)
                result.add_(Q_power, alpha=sign)
    
    return result

def test_accuracy(batch_size=4, matrix_size=128, num_terms=5):
    """Test if both approaches produce the same result"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random skew-symmetric matrices
    Q_raw = torch.randn(batch_size, matrix_size, matrix_size, device=device)
    Q = create_skew_symmetric(Q_raw)
    
    # Compute using both approaches
    result1 = cayley_approach1(Q, num_terms)
    result2 = cayley_approach2(Q, num_terms)
    
    # Check difference
    diff = torch.abs(result1 - result2).max().item()
    print(f"Maximum absolute difference between approaches: {diff:.8e}")
    
    # Also compare with direct computation using matrix inverse for reference
    I = torch.eye(matrix_size, device=device).expand(batch_size, matrix_size, matrix_size)
    direct = torch.bmm(I - Q, torch.inverse(I + Q))
    
    diff1 = torch.abs(result1 - direct).max().item()
    diff2 = torch.abs(result2 - direct).max().item()
    
    print(f"Approach 1 vs direct: {diff1:.8e}")
    print(f"Approach 2 vs direct: {diff2:.8e}")
    
    return diff < 1e-5

def benchmark_approaches(batch_sizes=[1], 
                         matrix_sizes=[256, 512, 1024], 
                         num_terms=5,
                         num_runs=50,
                         warmup=10):
    """Benchmark both approaches across different configurations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {
        'batch_size': [],
        'matrix_size': [],
        'approach1_time': [],
        'approach2_time': [],
        'speedup': []
    }
    
    for batch_size in batch_sizes:
        for matrix_size in matrix_sizes:
            print(f"Testing batch_size={batch_size}, matrix_size={matrix_size}")
            
            # Create random skew-symmetric matrices
            Q_raw = torch.randn(batch_size, matrix_size, matrix_size, device=device)
            Q = create_skew_symmetric(Q_raw)
            
            # Warmup
            for _ in range(warmup):
                _ = cayley_approach1(Q, num_terms)
                _ = cayley_approach2(Q, num_terms)
            
            # Time approach 1
            sync_cuda()
            start = time.time()
            for _ in range(num_runs):
                _ = cayley_approach1(Q, num_terms)
            sync_cuda()
            approach1_time = (time.time() - start) / num_runs * 1000  # ms
            
            # Time approach 2
            sync_cuda()
            start = time.time()
            for _ in range(num_runs):
                _ = cayley_approach2(Q, num_terms)
            sync_cuda()
            approach2_time = (time.time() - start) / num_runs * 1000  # ms
            
            # Calculate speedup (approach1 / approach2)
            speedup = approach1_time / approach2_time if approach2_time > 0 else float('inf')
            
            # Store results
            results['batch_size'].append(batch_size)
            results['matrix_size'].append(matrix_size)
            results['approach1_time'].append(approach1_time)
            results['approach2_time'].append(approach2_time)
            results['speedup'].append(speedup)
            
            print(f"  Approach 1: {approach1_time:.3f} ms")
            print(f"  Approach 2: {approach2_time:.3f} ms")
            print(f"  Speedup (Approach1/Approach2): {speedup:.2f}x")
            print(f"  {'Approach 2 is faster' if speedup > 1 else 'Approach 1 is faster'}")
    
    return results

def plot_results(results):
    """Plot the benchmark results"""
    plt.figure(figsize=(15, 10))
    
    # Plot by matrix size
    plt.subplot(2, 1, 1)
    for matrix_size in sorted(set(results['matrix_size'])):
        batch_sizes = []
        speedups = []
        
        for i in range(len(results['matrix_size'])):
            if results['matrix_size'][i] == matrix_size:
                batch_sizes.append(results['batch_size'][i])
                speedups.append(results['speedup'][i])
        
        plt.plot(batch_sizes, speedups, 'o-', label=f'Matrix size {matrix_size}')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.title('Speedup by Matrix Size (Approach1/Approach2)')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor')
    plt.grid(True)
    plt.legend()
    
    # Plot by batch size
    plt.subplot(2, 1, 2)
    for batch_size in sorted(set(results['batch_size'])):
        matrix_sizes = []
        speedups = []
        
        for i in range(len(results['batch_size'])):
            if results['batch_size'][i] == batch_size:
                matrix_sizes.append(results['matrix_size'][i])
                speedups.append(results['speedup'][i])
        
        plt.plot(matrix_sizes, speedups, 'o-', label=f'Batch size {batch_size}')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.title('Speedup by Batch Size (Approach1/Approach2)')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Factor')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cayley_approaches_comparison.png')
    plt.show()

def test_num_terms_impact(batch_size=8, matrix_size=256, max_terms=10):
    """Test the impact of number of terms on accuracy and performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random skew-symmetric matrices
    Q_raw = torch.randn(batch_size, matrix_size, matrix_size, device=device)
    Q = create_skew_symmetric(Q_raw)
    
    # Compute reference using direct matrix inverse
    I = torch.eye(matrix_size, device=device).expand(batch_size, matrix_size, matrix_size)
    direct = torch.bmm(I - Q, torch.inverse(I + Q))
    
    terms = list(range(1, max_terms + 1))
    approach1_errors = []
    approach2_errors = []
    approach1_times = []
    approach2_times = []
    
    for num_terms in terms:
        print(f"Testing with {num_terms} terms")
        
        # Measure approach 1
        sync_cuda()
        start = time.time()
        result1 = cayley_approach1(Q, num_terms)
        sync_cuda()
        approach1_time = (time.time() - start) * 1000  # ms
        
        # Measure approach 2
        sync_cuda()
        start = time.time()
        result2 = cayley_approach2(Q, num_terms)
        sync_cuda()
        approach2_time = (time.time() - start) * 1000  # ms
        
        # Compute errors
        error1 = torch.abs(result1 - direct).max().item()
        error2 = torch.abs(result2 - direct).max().item()
        
        approach1_errors.append(error1)
        approach2_errors.append(error2)
        approach1_times.append(approach1_time)
        approach2_times.append(approach2_time)
        
        print(f"  Approach 1: Error={error1:.8e}, Time={approach1_time:.3f}ms")
        print(f"  Approach 2: Error={error2:.8e}, Time={approach2_time:.3f}ms")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot errors
    plt.subplot(2, 1, 1)
    plt.semilogy(terms, approach1_errors, 'o-', label='Approach 1')
    plt.semilogy(terms, approach2_errors, 's-', label='Approach 2')
    plt.title('Error vs Number of Terms')
    plt.xlabel('Number of Terms')
    plt.ylabel('Maximum Absolute Error')
    plt.grid(True)
    plt.legend()
    
    # Plot times
    plt.subplot(2, 1, 2)
    plt.plot(terms, approach1_times, 'o-', label='Approach 1')
    plt.plot(terms, approach2_times, 's-', label='Approach 2')
    plt.title('Computation Time vs Number of Terms')
    plt.xlabel('Number of Terms')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cayley_terms_comparison.png')
    plt.show()

def main():
    print("Testing Cayley Transform Approaches")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test accuracy
    print("\nTesting accuracy...")
    is_accurate = test_accuracy()
    print(f"Approaches match: {is_accurate}")
    
    # Benchmark approaches
    print("\nBenchmarking approaches...")
    results = benchmark_approaches(
        batch_sizes=[16],
        matrix_sizes=[64, 128, 256, 512],
        num_terms=5,
        num_runs=20
    )
    
    # Plot results
    plot_results(results)
    
    # Test impact of number of terms
    print("\nTesting impact of number of terms...")
    test_num_terms_impact(batch_size=4, matrix_size=256, max_terms=10)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()