# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector


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
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    def cayley(self, w):
        """Convert skew-symmetric matrix to orthogonal matrix using Cayley transform."""
        if w.dtype == torch.bfloat16:
            w = w.to(torch.float32)
            skew = 0.5 * (w - w.t())
            I = torch.eye(w.shape[0], device=w.device)
            Q = torch.mm(I + skew, torch.inverse(I - skew))
            Q = Q.to(torch.bfloat16)
        else:
            skew = 0.5 * (w - w.t())
            I = torch.eye(w.shape[0], device=w.device)
            Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q

    def cayley_backward(self, grad_R, soft_S):
        """Compute gradient w.r.t skew-symmetric parameters."""
        if grad_R.dtype == torch.bfloat16:
            grad_R = grad_R.to(torch.float32)
            skew = 0.5 * (grad_R - grad_R.t())
            I = torch.eye(grad_R.shape[0], device=grad_R.device)
            grad_S = torch.mm(I + skew, soft_S)
            grad_S = grad_S.to(torch.bfloat16)
        else:
            skew = 0.5 * (grad_R - grad_R.t())
            I = torch.eye(grad_R.shape[0], device=grad_R.device)
            grad_S = torch.mm(I + skew, soft_S)
        return grad_S

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

                # SOFT handling
                if "rank" in group:
                    if "soft_S" not in state:
                        r = group["rank"]
                        if p.dtype == torch.bfloat16:
                            state['soft_S'] = torch.zeros(r, r, dtype=torch.bfloat16, device=p.device)
                        else:
                            state['soft_S'] = torch.zeros(r, r, device=p.device)
                        state['indices'] = None

                    # Update indices periodically - always sample from in_features dimension
                    if state['indices'] is None or state["step"] % group["update_proj_gap"] == 0:
                        in_features = p.shape[1]  # Weight shape is (out_features, in_features)
                        state['indices'] = torch.randperm(in_features, device=p.device)[:group["rank"]]

                    if len(p.shape) > 1:
                        indices = state['indices']
                        # Select columns (in_features dimensions)
                        weight_subset = p[:, indices]
                        # Select corresponding gradient dimensions
                        grad_subset = grad[:, indices]
                        grad_R = grad_subset @ weight_subset.t()

                        breakpoint()
                        
                        # Compute gradient w.r.t skew-symmetric parameters
                        grad_S = self.cayley_backward(grad_R, state['soft_S'])
                        
                        # Update skew-symmetric parameters
                        state['soft_S'].add_(grad_S, alpha=group['lr'])
                        
                        # Convert to orthogonal matrix using Cayley
                        Q = self.cayley(state['soft_S'])
                        
                        # Right multiply with Q to rotate input features
                        p[:, indices] = weight_subset @ Q

                        # Reset periodically
                        if state["step"] % group["update_proj_gap"] == 0:
                            state['soft_S'].zero_()
                            state['indices'] = None

                    continue  # Skip regular AdamW update for SOFT parameters

                # Regular AdamW update for non-SOFT parameters
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

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

                p.addcdiv_(exp_avg, denom, value=-step_size)

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


import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F


class CayleyAdamW_working_great_but_is_not_what_we_want(Optimizer):
    """
    Implements AdamW algorithm with Cayley transform for orthogonal rotations.
    
    For parameter groups with 'soft_rank' attribute, it applies the Cayley transformation
    to create orthogonal rotations of the weights. For other parameter groups, it uses
    standard AdamW updates.
    
    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            To apply Cayley transformations, add 'soft_rank' to the parameter group.
        lr (float, optional): Learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): Coefficients for computing running averages
            of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): Term added to denominator for numerical stability (default: 1e-6)
        weight_decay (float, optional): Weight decay coefficient (default: 0.0)
        correct_bias (bool, optional): Correct bias in Adam (default: True)
        no_deprecation_warning (bool, optional): Disable deprecation warning (default: False)
    
    Example:
        >>> model = YourModel()
        >>> # Group parameters - apply Cayley transform only to specific linear layers
        >>> optimizer_grouped_parameters = [
        ...     {
        ...         "params": [p for n, p in model.named_parameters() if "linear1" in n or "linear2" in n],
        ...         "soft_rank": 128,  # Apply Cayley transform with rank 128
        ...         "soft_update_freq": 50,  # Update actual weights every 50 steps
        ...     },
        ...     {
        ...         "params": [p for n, p in model.named_parameters() if not ("linear1" in n or "linear2" in n)],
        ...     },
        ... ]
        >>> optimizer = CayleyAdamW(optimizer_grouped_parameters, lr=1e-4)
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
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation is based on AdamW from Hugging Face Transformers",
                FutureWarning,
            )
        
        # Validate parameters
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
        }
        super().__init__(params, defaults)
        
        # Initialize Cayley-specific state for parameters with 'soft_rank'
        for group in self.param_groups:
            if "soft_rank" in group:
                for p in group["params"]:
                    if p.dim() != 2:
                        warnings.warn(
                            f"Parameter with shape {p.shape} found in group with 'soft_rank'. "
                            "Cayley transform only works with 2D parameters (weight matrices)."
                        )
                        continue
                        
                    # Get state dict for this parameter
                    state = self.state[p]
                    
                    # Set default update frequency if not specified
                    update_freq = group.get("update_indices_gap", 50)
                    
                    # Initialize Cayley-specific state
                    rank = min(group["soft_rank"], p.shape[1])
                    state["original_weight"] = p.data.clone()
                    state["Q"] = torch.zeros(rank, rank, device=p.device, dtype=p.dtype)
                    state["exp_avg_Q"] = torch.zeros_like(state["Q"])
                    state["exp_avg_sq_Q"] = torch.zeros_like(state["Q"])
                    state["selected_indices"] = torch.arange(p.shape[1], device=p.device)[:rank]
                    state["steps_since_update"] = 0
                    
                    # Regular Adam state
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

    def get_cayley_transform(self, Q):
        """Compute orthogonal matrix R from Q using Cayley transform."""
        # Cast to float32 for stable matrix inversion
        Q_float32 = Q.to(torch.float32)
        
        # Make Q skew-symmetric: Q = (Q - Q.t())/2
        Q_skew = (Q_float32 - Q_float32.t()) / 2
        
        # Cayley transform: R = (I + Q)(I - Q)^(-1)
        I = torch.eye(Q.shape[0], device=Q.device, dtype=torch.float32)
        R = torch.matmul(I + Q_skew, torch.linalg.inv(I - Q_skew))
        
        # Cast back to original dtype
        return R.to(Q.dtype)
    
    def update_indices(self, p, state, rank):
        """Update the selected indices randomly."""
        with torch.no_grad():
            state["selected_indices"] = torch.randperm(p.shape[1], device=p.device)[:rank]

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

                # State initialization if not already done
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                # Standard AdamW state update
                state["step"] += 1
                
                # Check if this parameter should use Cayley transform
                use_cayley = "soft_rank" in group and p.dim() == 2 and "Q" in state
                
                if use_cayley:
                    # Increment counter for Cayley updates
                    state["steps_since_update"] += 1
                    
                    # Get Cayley-specific state
                    Q = state["Q"]
                    selected_indices = state["selected_indices"]
                    rank = Q.shape[0]
                    
                    # Compute Q gradient (approximate)
                    # Here we're approximating the gradient for Q based on the gradient for the weights
                    selected_grad = grad[:, selected_indices]
                    R = self.get_cayley_transform(Q)
                    
                    # This is a simplified gradient computation for Q
                    # The actual gradient would involve a more complex computation involving the Cayley transform
                    original_weight = state["original_weight"][:, selected_indices]
                    Q_grad = torch.matmul(selected_grad.t(), original_weight)
                    # Make the gradient skew-symmetric to ensure Q stays skew-symmetric
                    Q_grad = (Q_grad - Q_grad.t()) / 2
                    
                    # Update Q with AdamW
                    exp_avg_Q, exp_avg_sq_Q = state["exp_avg_Q"], state["exp_avg_sq_Q"]
                    beta1, beta2 = group["betas"]
                    
                    # Update biased first moment estimate
                    exp_avg_Q.mul_(beta1).add_(Q_grad, alpha=1.0 - beta1)
                    
                    # Update biased second raw moment estimate
                    exp_avg_sq_Q.mul_(beta2).addcmul_(Q_grad, Q_grad, value=1.0 - beta2)
                    
                    # Compute bias-corrected estimates
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    
                    # Compute step size
                    step_size = group["lr"]
                    if group["correct_bias"]:
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                    
                    # Update Q
                    denom = exp_avg_sq_Q.sqrt().add_(group["eps"])
                    Q.addcdiv_(exp_avg_Q, denom, value=-step_size)
                    
                    # Make Q skew-symmetric explicitly
                    Q.copy_((Q - Q.t()) / 2)
                    
                    # Apply weight decay to Q if needed
                    if group["weight_decay"] > 0.0:
                        Q.add_(Q, alpha=(-group["lr"] * group["weight_decay"]))
                    
                    # Periodically apply the transform to actual weights
                    update_freq = group.get("soft_update_freq", 50)
                    if state["steps_since_update"] >= update_freq:
                        # Compute the orthogonal matrix R
                        R = self.get_cayley_transform(Q)
                        
                        # Apply R to the selected columns of the weight matrix
                        original_weight = state["original_weight"]
                        selected_weights = original_weight[:, selected_indices].to(R.dtype)
                        transformed_selected = torch.matmul(selected_weights, R.t()).to(original_weight.dtype)
                        
                        # Update the actual weight matrix
                        p.data[:, selected_indices] = transformed_selected
                        
                        # Update stored original weight
                        state["original_weight"] = p.data.clone()
                        
                        # Reset Q and counters
                        Q.zero_()
                        state["exp_avg_Q"].zero_()
                        state["exp_avg_sq_Q"].zero_()
                        state["steps_since_update"] = 0
                        
                        # Optionally update selected indices
                        if group.get("soft_random_indices", False):
                            self.update_indices(p, state, rank)
                
                # Also apply standard AdamW update 
                # (for non-Cayley params or alongside Cayley for better convergence)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Compute step size
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply standard AdamW update
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                
                # If using Cayley, only update non-selected indices with AdamW
                if use_cayley:
                    # Create a mask for non-selected indices
                    mask = torch.ones(p.shape[1], dtype=torch.bool, device=p.device)
                    mask[state["selected_indices"]] = False
                    
                    # Only apply standard update to non-selected indices
                    # Create a temporary view for updating
                    temp_p = p.clone()
                    temp_p.addcdiv_(exp_avg, denom, value=-step_size)
                    
                    # Apply weight decay
                    if group["weight_decay"] > 0.0:
                        temp_p.add_(temp_p, alpha=(-group["lr"] * group["weight_decay"]))
                    
                    # Update only the non-selected indices
                    p.data[:, mask] = temp_p[:, mask]
                else:
                    breakpoint()
                    # Standard update for regular parameters
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    
                    # Apply weight decay
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class CayleyAdamW(Optimizer):
    """
    Implements AdamW algorithm with Cayley transform for orthogonal rotations.
    
    For parameter groups with 'soft_rank' attribute, it applies the Cayley transformation
    to create orthogonal rotations of the weights. For other parameter groups, it uses
    standard AdamW updates.
    
    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            To apply Cayley transformations, add 'soft_rank' to the parameter group.
        lr (float, optional): Learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): Coefficients for computing running averages
            of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): Term added to denominator for numerical stability (default: 1e-6)
        weight_decay (float, optional): Weight decay coefficient (default: 0.0)
        correct_bias (bool, optional): Correct bias in Adam (default: True)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        # Validate parameters
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
        }
        super().__init__(params, defaults)
        
        # Initialize Cayley-specific state for parameters with 'soft_rank'
        for group in self.param_groups:
            if "soft_rank" in group:
                for p in group["params"]:
                    if p.dim() != 2:
                        warnings.warn(
                            f"Parameter with shape {p.shape} found in group with 'soft_rank'. "
                            "Cayley transform only works with 2D parameters (weight matrices)."
                        )
                        continue
                        
                    # Get state dict for this parameter
                    state = self.state[p]
                    
                    # Initialize Cayley-specific state
                    state["original_weight"] = p.data.clone()
                    
                    # Use in_features x in_features for Q (the input dimension of the weight matrix)
                    in_features = p.shape[1]
                    state["Q"] = torch.zeros(in_features, in_features, device=p.device, dtype=p.dtype)
                    state["exp_avg_Q"] = torch.zeros_like(state["Q"])
                    state["exp_avg_sq_Q"] = torch.zeros_like(state["Q"])
                    state["step"] = 0
                    
                    # Flag to indicate this parameter uses Cayley transform
                    state["uses_cayley"] = True

    def get_cayley_transform(self, Q):
        """Compute orthogonal matrix R from Q using Cayley transform."""
        # Cast to float32 for stable matrix inversion
        Q_float32 = Q.to(torch.float32)
        
        # Make Q skew-symmetric: Q = (Q - Q.t())/2
        Q_skew = (Q_float32 - Q_float32.t()) / 2
        
        # Cayley transform: R = (I + Q)(I - Q)^(-1)
        I = torch.eye(Q.shape[0], device=Q.device, dtype=torch.float32)
        try:
            # Numerically more stable than direct inversion
            # R = (I + Q)(I - Q)^(-1)
            iminusq_inv = torch.linalg.solve(I - Q_skew, I)  # (I - Q)^(-1)
            R = torch.matmul(I + Q_skew, iminusq_inv)
        except RuntimeError:
            # If solve fails, add a small epsilon to diagonal
            iminusq_inv = torch.linalg.solve(I - Q_skew + 1e-7 * I, I)
            R = torch.matmul(I + Q_skew, iminusq_inv)
        
        # Cast back to original dtype
        return R.to(Q.dtype), iminusq_inv.to(Q.dtype)

    def compute_q_gradient(self, weight_grad, W, R, iminusq_inv, Q):
        """
        Compute an accurate gradient for Q based on the Cayley transform derivative.
        
        This computes dL/dQ where L is the loss and Q is the skew-symmetric matrix
        that parameterizes the orthogonal matrix R through the Cayley transform.
        
        Args:
            weight_grad: Gradient of loss with respect to weight matrix (dL/dW_eff)
            W: Original weight matrix
            R: Orthogonal matrix from Cayley transform
            iminusq_inv: (I - Q)^(-1) from the Cayley transform calculation
            Q: Current Q matrix
            
        Returns:
            Gradient for Q
        """
        # weight_grad has shape (out_features, in_features)
        # W has shape (out_features, in_features)
        # R, iminusq_inv, Q all have shape (in_features, in_features)
        
        # First, compute R_grad = dL/dR which represents how R should change to decrease the loss
        # For W_eff = W * R^T, we have dL/dR = (dL/dW_eff * W)^T
        R_grad = torch.matmul(weight_grad.t(), W)  # (in_features, in_features)
        
        # Make R_grad skew-symmetric to ensure it produces valid updates to Q
        R_grad_skew = (R_grad - R_grad.t()) / 2
        
        # For the Cayley transform R = (I + Q)(I - Q)^(-1),
        # the gradient dR/dQ involves iminusq_inv = (I - Q)^(-1)
        # The exact formula is complex, but we can use a good approximation:
        
        # Use intermediate results from Cayley transform to compute gradient
        # This formula comes from matrix calculus rules applied to the Cayley transform
        I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
        
        # Compute the gradient for Q using the chain rule through R
        # dL/dQ = dL/dR * dR/dQ
        # For skew-symmetric Q, the derivative has a special form
        
        # The gradient for Q can be approximated as:
        # G = 2 * (I - Q)^(-T) * R_grad * (I + Q)^T
        temp = torch.matmul(R_grad_skew, (I + Q).t())
        Q_grad = 2 * torch.matmul(iminusq_inv.t(), temp)
        
        # Ensure Q_grad is skew-symmetric
        # Q_grad = (Q_grad - Q_grad.t()) / 2
        
        return Q_grad

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

                # Check if this parameter should use Cayley transform
                use_cayley = "soft_rank" in group and p.dim() == 2 and state.get("uses_cayley", False)
                
                if use_cayley:
                    # Update using Cayley transform
                    
                    # Increment step counter
                    state["step"] += 1
                    
                    # Get Cayley-specific state
                    Q = state["Q"]
                    original_weight = state["original_weight"]
                    
                    # Get current orthogonal matrix R and (I-Q)^(-1) for gradient computation
                    R, iminusq_inv = self.get_cayley_transform(Q)
                    
                    # Compute accurate gradient for Q
                    Q_grad = self.compute_q_gradient(grad, original_weight, R, iminusq_inv, Q)
                    
                    # Update Q with AdamW
                    exp_avg_Q, exp_avg_sq_Q = state["exp_avg_Q"], state["exp_avg_sq_Q"]
                    beta1, beta2 = group["betas"]
                    
                    # Update biased first moment estimate
                    exp_avg_Q.mul_(beta1).add_(Q_grad, alpha=1.0 - beta1)
                    
                    # Update biased second raw moment estimate
                    exp_avg_sq_Q.mul_(beta2).addcmul_(Q_grad, Q_grad, value=1.0 - beta2)
                    
                    # Compute bias-corrected estimates
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    
                    # Compute step size
                    step_size = group["lr"]
                    if group["correct_bias"]:
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                    
                    # Update Q
                    denom = exp_avg_sq_Q.sqrt().add_(group["eps"])
                    Q.addcdiv_(exp_avg_Q, denom, value=-step_size)
                    
                    # Ensure Q remains skew-symmetric
                    # Q.copy_((Q - Q.t()) / 2)
                    
                    # Apply weight decay to Q if needed
                    if group["weight_decay"] > 0.0:
                        decay_factor = 1.0 - group["lr"] * group["weight_decay"]
                        Q.mul_(decay_factor)
                    
                    # Apply the transform at every step
                    R, _ = self.get_cayley_transform(Q)
                    
                    # Apply R to the original weight matrix
                    # original_weight has shape (out_features, in_features)
                    # R has shape (in_features, in_features)
                    transformed_weight = torch.matmul(original_weight, R.t()).to(original_weight.dtype)
                    
                    # Update the actual weight matrix
                    p.data.copy_(transformed_weight)
                    
                else:
                    # Standard AdamW update for non-Cayley parameters
                    
                    # State initialization if not already done
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    # Standard AdamW state update
                    state["step"] += 1
                    
                    # Standard AdamW update
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    
                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    
                    # Compute step size
                    step_size = group["lr"]
                    if group["correct_bias"]:
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                    
                    # Update parameter
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    
                    # Apply weight decay
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss