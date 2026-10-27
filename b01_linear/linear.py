import torch

def linear_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      (B, in_features)
    weight: (out_features, in_features)
    bias:   (out_features,)

    返回 y: (B, out_features)
    """
    
    y = x @ weight.T + bias
    
    return y

def linear_backward(grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    grad_output: (B, out_features)

    返回：
    grad_input:  与 x 同形
    grad_weight: 与 weight 同形
    grad_bias:   (out_features,)
    """
    
    grad_input = grad_output @ weight
    grad_weight = grad_output.T @ x
    grad_bias = grad_output.sum(dim=0)
    
    return grad_input, grad_weight, grad_bias