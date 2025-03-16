import torch

def test_gpus():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Loop through each GPU and test it
    for i in range(num_gpus):
        print(f"\nTesting GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Set the device to the current GPU
        device = torch.device(f"cuda:{i}")
        
        # Create a simple tensor and perform a computation
        try:
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            
            # Check if the computation was successful
            if z.is_cuda:
                print(f"GPU {i} is functioning correctly.")
            else:
                print(f"GPU {i} failed to perform computation.")
        except Exception as e:
            print(f"GPU {i} encountered an error: {e}")

if __name__ == "__main__":
    test_gpus()