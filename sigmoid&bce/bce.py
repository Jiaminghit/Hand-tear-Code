import numpy as np
from sigmoid import *

def bce_probability(probabilities, targets, reduction="mean"):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    check_bce_shapes(probabilities, targets)
    positive = -targets * np.log(probabilities)
    negative = -(1 - targets) * np.log(1 - probabilities)
    loss = positive + negative
    if(reduction == "none"):
        return loss
    elif(reduction == "sum"):
        return np.sum(loss)
    elif(reduction =="mean"):
        return np.mean(loss)
    else:
        raise ValueError(f"不支持的 reduction：{reduction}")
    
def bce_backward(grad_output, probabilities, targets, reduction = "mean"):
    grad_output = np.asarray(grad_output, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    check_bce_shapes(probabilities, targets)
    grad_probabilities = -targets / probabilities + (1 - targets) / (1 - probabilities)
    if(reduction == "mean"):
        return grad_output * grad_probabilities / probabilities.size
    elif(reduction == "sum" or reduction == "none"):
        return grad_output * grad_probabilities
    else :
        raise ValueError(f"不支持的 reduction：{reduction}")
        
def bce_logits(logits, targets, reduction = "mean"):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    check_bce_shapes(logits, targets)
    losses = np.maximum(logits, 0) - targets * logits + np.log1p(np.exp(-np.abs(logits)))
    if(reduction == "none"):
        return losses
    elif(reduction == "sum"):
        return np.sum(losses)
    elif(reduction =="mean"):
        return np.mean(losses)
    else:
        raise ValueError(f"不支持的 reduction：{reduction}")  

def bce_logits_backward(grad_output, logits, targets, reduction = "mean"):
    grad_output = np.asarray(grad_output, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64) 
    check_bce_shapes(logits, targets)
    p = sigmoid(logits)
    grad_logits = p - targets
    if(reduction == "none" or reduction == "sum"):
        return grad_output * grad_logits
    elif(reduction =="mean"):
        return grad_output * grad_logits / logits.size
    else:
        raise ValueError(f"不支持的 reduction：{reduction}")  
    
def check_bce_shapes(predictions, targets):
    if(predictions.shape != targets.shape):
        raise ValueError(f"{predictions.shape}与{targets.shape}形状不符")
# probabilities = [0.8, 0.8, 0.5]
# targets       = [1,   0,   1]
# grad_output   = 1
# print(bce_probability(probabilities, targets, "none"))
# print(bce_backward(grad_output, probabilities, targets, "sum"))
# print(bce_backward(grad_output, probabilities, targets, "mean"))

# z = [-2, 0, 2]
# y = [0, 1, 1]
# p = sigmoid(z)
# grad_p = bce_backward(1, p, y, "mean")
# grad_z = sigmoid_backward(grad_p, p)
# print(grad_z)

# grad_p = bce_backward(1, p, y, "sum")
# grad_z = sigmoid_backward(grad_p, p)
# print(grad_z)

logits  = [-1000, 0, 1000]
targets = [1,     0, 0]
grad_output = 1
print(bce_logits_backward(grad_output, logits, targets, "mean"))