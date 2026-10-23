import numpy as np

def huber_loss(predictions, targets, delta=1.0, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    
    # check
    if not predictions.shape == targets.shape:
        raise ValueError(f"invalid {predictions} or {targets}")
    if delta < 0.0:
        raise ValueError(f"invalid delta {delta}")
    
    # loss
    res = predictions - targets
    res_abs = np.abs(res)
    mse_loss = res ** 2 / 2
    mae_loss = delta * (res_abs - delta / 2)
    huber_loss = np.where(res_abs <= delta, mse_loss, mae_loss)
    
    # output_check
    if reduction == "none":
        return huber_loss
    elif reduction == "sum":
        return np.sum(huber_loss)
    elif reduction == "mean":
        return np. mean(huber_loss)
    else :
        raise ValueError(f"invalid reduction {reduction}")

def smooth_l1_loss(predictions, targets, beta=1.0, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    
    # check
    if not predictions.shape == targets.shape:
        raise ValueError(f"invalid {predictions} or {targets}")
    if beta < 0.0:
        raise ValueError(f"invalid delta {beta}")
    
    # loss
    res = predictions - targets
    res_abs = np.abs(res)
    if beta == 0:
        smooth_l1_loss = res_abs
    else:
        mse_loss = res ** 2 / (2 * beta)
        mae_loss = res_abs - beta / 2
        smooth_l1_loss = np.where(res_abs <= beta, mse_loss, mae_loss)  

    # output_check
    if reduction == "none":
        return smooth_l1_loss
    elif reduction == "sum":
        return np.sum(smooth_l1_loss)
    elif reduction == "mean":
        return np. mean(smooth_l1_loss)
    else :
        raise ValueError(f"invalid reduction {reduction}")

def huber_backward(grad_output, predictions, targets, delta=1.0, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    
    # check
    if not predictions.shape == targets.shape:
        raise ValueError(f"invalid {predictions} or {targets}")
    if delta < 0.0:
        raise ValueError(f"invalid delta {delta}")
    
    # grad_input
    res = predictions - targets
    grad_input = np.clip(res, -delta, delta)
    
    # grad_output
    if reduction == "none" or reduction == "sum":
        grad_predictions = grad_output * grad_input
    elif reduction == "mean":
        grad_predictions = grad_output * grad_input / predictions.size
    else :
        raise ValueError(f"invalid {reduction}")
    return grad_predictions 
    
def smooth_l1_backward(grad_output, predictions, targets, beta=1.0, reduction="mean"):
    # arraization
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    # check
    if not predictions.shape == targets.shape:
        raise ValueError(f"invalid {predictions} or {targets}")
    if beta < 0.0:
        raise ValueError(f"invalid delta {beta}")    
        
    # grad_input
    res = predictions - targets
    if beta == 0:
        grad_input = np.sign(res)
    else:
        grad_input = np.clip(res, -beta, beta) / beta
           
    # grad_output
    if reduction == "none" or reduction == "sum":
        grad_predictions = grad_output * grad_input
    elif reduction == "mean":
        grad_predictions = grad_output * grad_input / predictions.size
    else :
        raise ValueError(f"invalid {reduction}")
    return grad_predictions      
        
    

if __name__ == "__main__":
    predictions = [[2, 0], [3, 7]]
    targets = [[1, 2], [3, 4]]
    # print(huber_loss(predictions, targets, 1.0, "none"))
    # print(huber_loss(predictions, targets, 1.0, "sum"))
    # print(huber_loss(predictions, targets, 1.0, "mean"))
    
    # print(huber_loss(predictions, targets, 2.0, "none"))
    # print(huber_loss(predictions, targets, 2.0, "sum"))
    # print(huber_loss(predictions, targets, 2.0, "mean"))
    
    # print(huber_backward([[1, 2], [3, 4]], predictions, targets, 2.0, "none"))
    # print(huber_backward(1, predictions, targets, 2.0, "sum"))
    # print(huber_backward(1, predictions, targets, 2.0, "mean"))
    
    # print(smooth_l1_loss(predictions, targets, 2.0, "none"))
    # print(smooth_l1_loss(predictions, targets, 2.0, "sum"))
    # print(smooth_l1_loss(predictions, targets, 2.0, "mean"))    
    
    # print(smooth_l1_loss(predictions, targets, 0.0, "none"))
    # print(smooth_l1_loss(predictions, targets, 0.0, "sum"))
    # print(smooth_l1_loss(predictions, targets, 0.0, "mean"))      
    
    print(smooth_l1_backward([[1, 2], [3, 4]], predictions, targets, 2.0, "none"))
    print(smooth_l1_backward(1, predictions, targets, 2.0, "sum"))
    print(smooth_l1_backward(1, predictions, targets, 2.0, "mean"))    