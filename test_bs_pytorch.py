import torch
import time

def original_implementation(transformed_weight, R_left, R_right):
    R_left_bs = torch.block_diag(*list(R_left))
    R_right_bs = torch.block_diag(*list(R_right))
    temp = transformed_weight @ R_right_bs
    transformed_weight = R_left_bs @ temp
    return transformed_weight

def optimized_implementation_loop_based(transformed_weight, R_left, R_right, r, d_out, d_in): # Renamed to clarify it's loop-based and not always better
    temp = torch.zeros_like(transformed_weight)
    for j in range(r):
        temp[:, j*d_in:(j+1)*d_in] = transformed_weight[:, j*d_in:(j+1)*d_in] @ R_right[j]

    transformed_weight_optimized = torch.zeros_like(transformed_weight)
    for i in range(r):
        transformed_weight_optimized[i*d_out:(i+1)*d_out, :] = R_left[i] @ temp[i*d_out:(i+1)*d_out, :]
    return transformed_weight_optimized

def test_equivalence(r, d_out, d_in, optimized_impl, device="cpu", dtype=torch.float32, tolerance=1e-05): # Added tolerance argument
    out_features = r * d_out
    in_features = r * d_in

    transformed_weight_original = torch.randn(out_features, in_features, dtype=dtype, device=device)
    transformed_weight_optimized_input = transformed_weight_original.clone()
    R_left = torch.randn(r, d_out, d_out, dtype=dtype, device=device)
    R_right = torch.randn(r, d_in, d_in, dtype=dtype, device=device)

    original_output = original_implementation(transformed_weight_original, R_left, R_right)
    optimized_output = optimized_impl(transformed_weight_optimized_input, R_left, R_right, r, d_out, d_in)

    if not torch.allclose(original_output, optimized_output, atol=tolerance): # Added atol argument
        print("Equivalence test failed!")
        print(f"Implementation: {optimized_impl.__name__}")
        print("r=", r, "d_out=", d_out, "d_in=", d_in)
        print("Max difference:", torch.max(torch.abs(original_output - optimized_output)))
        print("Mean difference:", torch.mean(torch.abs(original_output - optimized_output)))
        assert False, "Equivalence test failed!"
    else:
        print(f"Equivalence test passed for: {optimized_impl.__name__}")

def test_time_comparison(r, d_out, d_in, optimized_impl, num_iterations=10, device="cpu", dtype=torch.float32): # Added optimized_impl argument
    out_features = r * d_out
    in_features = r * d_in

    transformed_weight_original = torch.randn(out_features, in_features, dtype=dtype, device=device)
    transformed_weight_optimized_input = transformed_weight_original.clone()
    R_left = torch.randn(r, d_out, d_out, dtype=dtype, device=device)
    R_right = torch.randn(r, d_in, d_in, dtype=dtype, device=device)

    start_time_original = time.time()
    for _ in range(num_iterations):
        original_implementation(transformed_weight_original.clone(), R_left, R_right)
    original_time = time.time() - start_time_original

    start_time_optimized = time.time()
    for _ in range(num_iterations):
        optimized_impl(transformed_weight_optimized_input.clone(), R_left, R_right, r, d_out, d_in) # Use passed optimized impl
    optimized_time = time.time() - start_time_optimized

    print(f"For r={r}, d_out={d_out}, d_in={d_in}:")
    print(f"  Original implementation time: {original_time:.4f} seconds")
    print(f"  {optimized_impl.__name__} time: {optimized_time:.4f} seconds") # Print impl name
    speedup = original_time / optimized_time
    print(f"  Speedup ({optimized_impl.__name__} vs Original): {speedup:.2f}x")
    return original_time, optimized_time, speedup

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    optimized_impl_to_test = optimized_implementation_loop_based

    # Equivalence tests
    test_equivalence(r=2, d_out=4, d_in=8, optimized_impl=optimized_impl_to_test, device=device, dtype=dtype, tolerance=1e-05)
    test_equivalence(r=4, d_out=32, d_in=48, optimized_impl=optimized_impl_to_test, device=device, dtype=dtype, tolerance=1e-05) # Test with higher tolerance
    test_equivalence(r=3, d_out=5, d_in=5, optimized_impl=optimized_impl_to_test, device=device, dtype=dtype, tolerance=1e-05)

    # Time comparison tests (no change needed here)
    print("\nTime Comparison Tests:")
    test_time_comparison(r=2, d_out=64, d_in=64, optimized_impl=optimized_impl_to_test, device=device, dtype=dtype)
    test_time_comparison(r=4, d_out=32, d_in=32, optimized_impl=optimized_impl_to_test, device=device, dtype=dtype)
    test_time_comparison(r=8, d_out=16, d_in=16, optimized_impl=optimized_impl_to_test, device=device, dtype=dtype)
    test_time_comparison(r=16, d_out=8, d_in=8, optimized_impl=optimized_impl_to_test, device=device, dtype=dtype)

    print("\n--- Conclusion ---")
    print("For general use, the `original_implementation` using `torch.block_diag` is recommended.")
    print("The `optimized_implementation_loop_based` can be faster in some cases (e.g., larger d_out, d_in, smaller r),")
    print("but it is often slower due to Python loop overhead, especially for smaller block sizes.")