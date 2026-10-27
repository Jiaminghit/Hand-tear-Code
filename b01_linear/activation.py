import torch

def relu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    x: 任意形状的浮点 Tensor，先限定为有限值。
    返回：与 x 同形的 ReLU 输出。
    """
    return torch.where(x > 0, x, 0.0)
    
    
def relu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    grad_output: 与 x 同形、同 dtype、同 device。
    x: 前向时的输入。

    返回：输入梯度；x == 0 时取 0。
    """
    return torch.where(x > 0, grad_output, 0.0)