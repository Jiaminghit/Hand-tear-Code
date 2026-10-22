import numpy as np
from softmax.softmax import softmax
from softmax.softmax import logsoftmax


def cross_entropy_single(logits, target):
    logits = np.asarray(logits, dtype=np.float64)
    logits_max = np.max(logits)
    sum_exp_logits = np.sum(np.exp(logits - logits_max))
    loss = np.log(sum_exp_logits) - (logits[target] - logits_max)
    return loss

#################################  IMPORTANT  ########################################
def cross_entropy(logits, targets, reduction="mean", label_smoothing=0.0, ignore_index = -100):
    # logits = (N, C) 
    # targets = (N,)
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    num_samples = logits.shape[0]
    
    # label_smoothing check
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(f"{label_smoothing} is invalid input")
    
    # mark ignore_index
    # valid_mask = [True, True, False, ...]
    valid_mask = targets != ignore_index
    num_valid = np.count_nonzero(valid_mask)
    
    # get LogSoftmax
    log_probs = logsoftmax(logits, axis=-1)
    
    # get loss_hard
    # loss_hard.shape = (N, )
    loss_hard = np.zeros(num_samples, dtype=np.float64)
    # valid_rows get valid row number by valid_mask
    valid_rows = np.arange(num_samples)[valid_mask]
    valid_targets = targets[valid_rows]
    loss_hard[valid_mask] = -log_probs[valid_rows, valid_targets]
    
    # get loss_uniform
    # loss_uniform.shape = (N, )
    loss_uniform = -np.mean(log_probs, axis=-1)
    loss_uniform[~valid_mask] = 0.0
    
    # loss = (N,)
    loss = (1 - label_smoothing) * loss_hard + label_smoothing * loss_uniform
    
    if (reduction == "none"):
        return loss
    elif (reduction == "sum"):
        return np.sum(loss, axis=0)
    elif (reduction == "mean"):
        if (num_valid==0):
           raise ValueError("No valid sample") 
        return np.sum(loss, axis=0) / num_valid
    else:
        raise ValueError(f"{reduction} is invalid input")
########################################################################################    


def cross_entropy_single_backward(logit, target):
    logit = np.asarray(logit, dtype=np.float64)
    # softmax
    max_logit = np.max(logit)
    shifted_logit = logit - max_logit
    exp_logit = np.exp(shifted_logit)
    possibilities = exp_logit / np.sum(exp_logit)
    grad = possibilities.copy()
    grad[target] -= 1
    return grad

def cross_entropy_backward(grad_output, logits, targets, reduction="mean", label_smoothing=0.0, ignore_index=-100):
    # logits.shape = (N, C)
    # targets.shape = (N,)
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    num_samples = logits.shape[0]
    num_classes = logits.shape[1]
    
    # label_smoothing check
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(f"{label_smoothing} is invalid input")
    
    # mark ignore_index
    # valid_mask = [True, True, False, ...]
    valid_mask = targets != ignore_index
    # num_valid
    num_valid = np.count_nonzero(valid_mask)
    # valid_rows & valid_targets
    valid_rows = np.arange(num_samples)[valid_mask]
    valid_targets = targets[valid_mask]
    
    # softmax
    possibilities = softmax(logits)
    
    # grad.shape = (N, C)
    grad = np.zeros_like(logits)
    grad[valid_rows, :] = possibilities[valid_rows, :]
    grad[valid_rows, :] -= (label_smoothing / num_classes)
    grad[valid_rows, valid_targets] -= (1.0 - label_smoothing)
    
    if (reduction == "none"):
        # grad_output = (N,)
        grad_output = grad_output.reshape(-1, 1)
        grad_logits = grad_output * grad
    elif (reduction == "sum"):
        # grad_out = ()
        grad_logits = grad_output * grad
    elif (reduction == "mean"):
        # grad_out = ()
        if (num_valid==0):
            raise ValueError("No valid sample") 
        grad_logits = grad_output * grad / num_valid
    else :
        raise ValueError(f"{reduction} is invalid input")
    return grad_logits
    
    
# print(cross_entropy_single([1, 2, 3], 2))
# print(cross_entropy_single([0, 0, 0], 1))
# print(cross_entropy_single([1000, -1000], 1))
# logits = [[1, 2, 3],
#           [1, 2, 3]]
# targets = [2, 0]
# print(cross_entry(logits, targets, "mean"))
if __name__ == "__main__":
    
    logits = [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]
    targets = [2, -100, 0]

    print(cross_entropy(
        logits, targets,
        reduction="none",
        label_smoothing=0.3,
    ))
    # 预期：[0.70760596 0.         2.10760596]

    print(cross_entropy(
        logits, targets,
        reduction="mean",
        label_smoothing=0.3,
    ))
    # 预期：1.4076059644 左右
