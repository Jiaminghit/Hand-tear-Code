import numpy as np

def cross_entropy_single(logits, target):
    logits = np.asarray(logits, dtype=np.float64)
    logits_max = np.max(logits)
    sum_exp_logits = np.sum(np.exp(logits - logits_max))
    loss = np.log(sum_exp_logits) - (logits[target] - logits_max)
    return loss

print(cross_entropy_single([1, 2, 3], 2))
print(cross_entropy_single([0, 0, 0], 1))
print(cross_entropy_single([1000, -1000], 1))