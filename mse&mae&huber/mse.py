import numpy as np

def mse_loss(predictions, targets, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    # check
    if predictions.shape != targets.shape:
        raise ValueError(f"invalid {predictions} or {targets}")
    # loss_mse
    loss_mse = (predictions - targets) ** 2
    # 
    if reduction == "none":
        return loss_mse
    elif reduction == "sum":
        loss_mse = np.sum(loss_mse)
    elif reduction == "mean":
        loss_mse = np.mean(loss_mse)
    else :
        raise ValueError(f"invalid {reduction}")
    return loss_mse

def mse_backward(grad_output, predictions, targets, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    # check
    if predictions.shape != targets.shape:
            raise ValueError(f"invalid {predictions} or {targets}")
    # grad_input
    grad_input = 2 * (predictions - targets)
    # grad_predictions
    if reduction == "none" or reduction == "sum":
        grad_predictions = grad_output * grad_input
    elif reduction == "mean":
        grad_predictions = grad_output * grad_input / predictions.size
    else :
        raise ValueError(f"invalid {reduction}")
    return grad_predictions

if __name__ == "__main__":
    predictions = [[2, 0],[3, 7]]
    targets = [[1, 2],[3, 4]]
    print(mse_loss(predictions, targets, "none"))
    print(mse_loss(predictions, targets, "sum"))
    print(mse_loss(predictions, targets, "mean"))
    
    print(mse_backward(1, predictions, targets, "sum"))
    print(mse_backward(1, predictions, targets, "mean"))
    print(mse_backward([[1, 2], [3, 4]], predictions, targets, "none"))
    
    