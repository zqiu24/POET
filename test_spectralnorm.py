import torch

# Define your matrix
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])

# Define a scalar
alpha = 2.0

# Scale the matrix
scaled_A = alpha * A

# Compute the spectral norms
spectral_norm_A = torch.linalg.norm(A, ord=2)
spectral_norm_scaled_A = torch.linalg.norm(scaled_A, ord=2)

print("Spectral norm of A:", spectral_norm_A.item())
print("Spectral norm of alpha * A:", spectral_norm_scaled_A.item())
print("Is it equal to |alpha| * spectral_norm(A)?", torch.isclose(spectral_norm_scaled_A, torch.tensor(abs(alpha) * spectral_norm_A)))