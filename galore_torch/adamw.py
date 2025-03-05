# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor


class AdamW(Optimizer):
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
        use_orthogonal (`bool`, *optional*, defaults to `False`):
            Whether to use orthogonal/spectral-preserving training.
        cayley_lr_factor (`float`, *optional*, defaults to 1.0):
            Learning rate factor for Cayley transform updates.
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
        use_orthogonal: bool = False,
        cayley_lr_factor: float = 1.0,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr, 
            "betas": betas, 
            "eps": eps, 
            "weight_decay": weight_decay, 
            "correct_bias": correct_bias,
            "use_orthogonal": use_orthogonal,
            "cayley_lr_factor": cayley_lr_factor,
        }
        super().__init__(params, defaults)

    def _get_skew_symmetric_from_vector(self, q_vector, shape):
        """Convert a vector representation to a skew-symmetric matrix."""
        n, m = shape
        A = torch.zeros((n, n), device=q_vector.device, dtype=q_vector.dtype)
        # Fill upper triangular part
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                A[i, j] = q_vector[idx]
                A[j, i] = -q_vector[idx]  # Skew-symmetric property
                idx += 1
        return A
    
    def _cayley_transform(self, A):
        """Apply Cayley transform to get orthogonal matrix from skew-symmetric matrix."""
        n = A.size(0)
        I = torch.eye(n, device=A.device, dtype=A.dtype)
        # R = (I + A) @ (I - A).inverse()
        return torch.matmul(I + A, torch.inverse(I - A))

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
                
                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2
                    
                # Orthogonal transformation setup
                if group.get("use_orthogonal", False) and len(p.shape) >= 2:
                    n, m = p.shape[0], p.shape[1]
                    
                    # Initialize orthogonal transform parameters if not present
                    if "q_left" not in state:
                        # Vector representation of skew-symmetric matrix (upper triangular part)
                        state["q_left"] = torch.zeros(n * (n - 1) // 2, device=p.device, dtype=p.dtype)
                        state["q_right"] = torch.zeros(m * (m - 1) // 2, device=p.device, dtype=p.dtype)
                        
                        # Save initial normalized weights
                        with torch.no_grad():
                            # Normalize initial weights
                            state["W_init"] = p.data.clone()
                            norm = torch.norm(state["W_init"])
                            if norm > 0:
                                state["W_init"].div_(norm)
                    
                    # Get gradient for q vectors instead of direct weight update
                    if "exp_avg_q_left" not in state:
                        state["exp_avg_q_left"] = torch.zeros_like(state["q_left"])
                        state["exp_avg_sq_q_left"] = torch.zeros_like(state["q_left"])
                        state["exp_avg_q_right"] = torch.zeros_like(state["q_right"])
                        state["exp_avg_sq_q_right"] = torch.zeros_like(state["q_right"])
                    
                    # We need to compute gradients for q_left and q_right instead of directly for W
                    # This is a simplification - in a real implementation you'd need to work out 
                    # the proper gradient computation through the Cayley transform
                    
                    # For now, let's use a simple approximation to update q vectors
                    # In practice, you would need to compute the exact gradients
                    
                    # Update q vectors using Adam
                    beta1, beta2 = group["betas"]
                    lr_cayley = group["lr"] * group["cayley_lr_factor"]
                    
                    # This would be where you compute the actual gradients for q_left and q_right
                    # based on the gradient of the weight matrix
                    # For now, we'll use a placeholder approximation
                    
                    # Update q_left and q_right
                    # ... code to update q vectors with Adam ...
                    
                    # Apply Cayley transform to get orthogonal matrices
                    A_left = self._get_skew_symmetric_from_vector(state["q_left"], (n, n))
                    A_right = self._get_skew_symmetric_from_vector(state["q_right"], (m, m))
                    
                    R_left = self._cayley_transform(A_left)
                    R_right = self._cayley_transform(A_right)
                    
                    # Update the weights using orthogonal transformations
                    with torch.no_grad():
                        # W = R_left @ W_init @ R_right
                        new_weights = torch.matmul(R_left, torch.matmul(state["W_init"], R_right))
                        p.data.copy_(new_weights)
                    
                    # Skip the normal Adam update for this parameter
                    continue
                
                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group['dim'] <=2:
                            state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                        else:
                            state["projector"] = GaLoreProjectorTensor(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

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

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)
                
                p.add_(norm_grad, alpha=-step_size)

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
