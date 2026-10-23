import numpy as np

def mae_loss(predictions, targets, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    # check
    if predictions.shape != targets.shape:
        raise ValueError(f"invalid {predictions} or {targets}")
    # loss_mse
    loss_mae = np.abs(predictions - targets)
    # 
    if reduction == "none":
        return loss_mae
    elif reduction == "sum":
        loss_mae = np.sum(loss_mae)
    elif reduction == "mean":
        loss_mae = np.mean(loss_mae)
    else :
        raise ValueError(f"invalid {reduction}")
    return loss_mae

def mae_backward(grad_output, predictions, targets, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    # check
    if predictions.shape != targets.shape:
        raise ValueError(f"invalid {predictions} or {targets}")
    # grad_input
    grad_input = np.sign(predictions - targets)
    
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
    
    print(mae_loss(predictions, targets, "none"))
    print(mae_loss(predictions, targets, "sum"))
    print(mae_loss(predictions, targets, "mean"))
    
    print(mae_backward(1, predictions, targets, "sum"))
    print(mae_backward(1, predictions, targets, "mean"))
    print(mae_backward([[1, 2], [3, 4]], predictions, targets, "none"))