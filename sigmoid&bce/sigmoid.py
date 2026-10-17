import numpy as np

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    
    mask = x >= 0
    result[mask] = 1 / (1 + np.exp(-x[mask]))
    exp_x = np.exp(x[~mask])
    result[~mask] = exp_x / (1 + exp_x)
    
    return result

def sigmoid_backward(grad_output, probabilities):
    grad_input = grad_output * probabilities * (1 - probabilities)
    return grad_input

# print(sigmoid([-2, 0, 2]))
# print(sigmoid([-1000, 0, 1000]))
# grad_output = [1, 2, 3]
# print(sigmoid_backward(grad_output, sigmoid([-2, 0, 2])))