from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class RegressionLoss(ABC):
    """统一管理逐元素回归损失的前向与手动反向。"""

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"不支持的 reduction：{reduction}")
        self.reduction = reduction

    # def _residual(
    #     self, predictions: ArrayLike, targets: ArrayLike
    # ) -> np.ndarray:
    #     """统一转换输入、检查形状并计算残差。"""
    #     predictions = np.asarray(predictions, dtype=np.float64)
    #     targets = np.asarray(targets, dtype=np.float64)

    #     if predictions.shape != targets.shape:
    #         raise ValueError("predictions 与 targets 必须同形")
    #     if predictions.size == 0:
    #         raise ValueError("本练习要求输入非空")

    #     return predictions - targets

    def _mask_residual(
        self, 
        predictions: ArrayLike, 
        targets: ArrayLike, 
        mask: ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        
        if predictions.shape != targets.shape:
            raise ValueError("predictions 与 targets 必须同形")
        if predictions.size == 0:
            raise ValueError("本练习要求输入非空")        
        
        if mask is None:
            valid = np.ones(predictions.shape, dtype=bool)
        else:
            valid = np.asarray(mask)
            if valid.dtype != np.bool_ or valid.shape != predictions.shape:
                raise ValueError("mask 必须为与输入同形的布尔数组")
        
        residual = predictions[valid] - targets[valid]
        return residual, valid
    
    def forward(
        self, 
        predictions: ArrayLike, 
        targets: ArrayLike,
        mask: ArrayLike | None = None, 
    ) -> np.ndarray | float:
        """计算逐元素损失，并按配置进行汇总。"""
        residual, valid = self._mask_residual(predictions, targets, mask)
        losses = np.zeros(valid.shape, dtype=np.float64)

        if residual.size > 0:
            losses[valid] = self._elementwise_loss(residual)
        
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return float(np.sum(losses))
        
        if residual.size == 0:
            raise ValueError("没有有效元素，无法求平均")
        return float(np.sum(losses) / residual.size)

    def backward(
        self,
        grad_output: ArrayLike,
        predictions: ArrayLike,
        targets: ArrayLike,
        mask: ArrayLike | None = None,
    ) -> np.ndarray:
        """结合局部梯度、汇总方式和上游梯度计算预测值梯度。"""
        residual, valid = self._mask_residual(predictions, targets, mask)
        grad_output = np.asarray(grad_output, dtype=np.float64)

        expected_shape = valid.shape if self.reduction == "none" else ()
        if grad_output.shape != expected_shape:
            raise ValueError(f"grad_output 的形状应为 {expected_shape}")

        grad = np.zeros(valid.shape, dtype=np.float64)
        
        if residual.size == 0:
            if self.reduction == "mean":    
                raise ValueError("没有有效元素，无法求平均")
        
        local_grad = self._elementwise_grad(residual)
        if self.reduction == "mean":   
            local_grad = local_grad / residual.size

        if self.reduction == "none":
            grad[valid] = grad_output[valid] * local_grad
        else:
            grad[valid] = grad_output * local_grad
        
        return grad

    @abstractmethod
    def _elementwise_loss(self, residual: np.ndarray) -> np.ndarray:
        """由子类提供逐元素损失公式。"""
        raise NotImplementedError

    @abstractmethod
    def _elementwise_grad(self, residual: np.ndarray) -> np.ndarray:
        """由子类提供对预测值的逐元素局部梯度。"""
        raise NotImplementedError

class MSELoss(RegressionLoss):
    """平方误差损失。"""

    def _elementwise_loss(self, residual: np.ndarray) -> np.ndarray:
        return residual ** 2

    def _elementwise_grad(self, residual: np.ndarray) -> np.ndarray:
        return 2 * residual

class MAELoss(RegressionLoss):
    """绝对误差损失，零残差处选择零次梯度。"""

    def _elementwise_loss(self, residual: np.ndarray) -> np.ndarray:
        return np.abs(residual)

    def _elementwise_grad(self, residual: np.ndarray) -> np.ndarray:
        return np.sign(residual)
    
class HuberLoss(RegressionLoss):
    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta 必须为有限的正数")
        self.delta = delta        
    def _elementwise_loss(self, residual):
        res_abs = np.abs(residual)
        mse_loss = residual ** 2 / 2
        mae_loss = self.delta * (res_abs - self.delta / 2)
        return np.where(res_abs <= self.delta, mse_loss, mae_loss)
    def _elementwise_grad(self, residual):
        return np.clip(residual, -self.delta, self.delta)
        
class SmoothL1Loss(RegressionLoss):
    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta 必须为有限的非负数")
        self.beta = beta        
    def _elementwise_loss(self, residual):
        res_abs = np.abs(residual)
        if self.beta == 0:
            return np.abs(residual)
        mse_loss = residual ** 2 / (2 * self.beta)
        mae_loss = res_abs - self.beta / 2
        return np.where(res_abs < self.beta, mse_loss, mae_loss)
    def _elementwise_grad(self, residual):
        if self.beta == 0:
            return np.sign(residual)
        return np.clip(residual, -self.beta, self.beta) / self.beta
    
if __name__ == "__main__":
    predictions = [[2, 0], [3, 7]]
    targets = [[1, 2], [3, 4]]

    criterion = MSELoss(reduction="mean")

    print(criterion.forward(predictions, targets))
    # 3.5

    print(criterion.backward(1.0, predictions, targets))
    # [[ 0.5, -1. ],
    #  [ 0. ,  1.5]]