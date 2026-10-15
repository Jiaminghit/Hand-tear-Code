import numpy as np

def softmax(x, axis = -1):
    """沿指定轴将输入分数转换为数值稳定的概率。

    Args:
        x: 有限实数输入，可转换为数组，归约轴不能为空。
        axis: 归一化轴，默认为最后一轴。

    Returns:
        与输入形状相同的 float64 概率数组，沿 axis 求和约为 1。
    """
    x = np.asarray(x, dtype=np.float64)
    # 减去每组最大值以避免指数上溢；保留归约维度便于广播。
    max_x = np.max(x, axis = axis, keepdims=True)
    shifted_x = x - max_x
    exp_x = np.exp(shifted_x)
    possibilities = exp_x / np.sum(exp_x, axis = axis, keepdims=True)
    return possibilities

def logsumexp(x, axis = -1):
    """沿指定轴计算数值稳定的指数和对数，并保留归约维度。

    Args:
        x: 有限实数输入，可转换为数组，归约轴不能为空。
        axis: 归约轴，默认为最后一轴。

    Returns:
        float64 数组；保留输入维数，归约轴的长度为 1。
    """
    x = np.asarray(x, dtype=np.float64)
    max_x = np.max(x, axis=axis, keepdims=True)
    shifted_x = x - max_x
    exp_x = np.exp(shifted_x)
    # 加回最大值，恢复原输入的 LogSumExp，补偿此前的平移。
    lse = max_x + np.log(np.sum(exp_x, axis=axis, keepdims=True))
    return lse

def logsoftmax(x, axis = -1):
    """沿指定轴直接计算数值稳定的对数概率。

    Args:
        x: 有限实数输入，可转换为数组，归约轴不能为空。
        axis: 归一化轴，默认为最后一轴。

    Returns:
        与输入形状相同的 float64 对数概率数组，取指数后沿 axis 求和约为 1。
    """
    x = np.asarray(x, dtype=np.float64)
    max_x = np.max(x, axis=axis, keepdims=True)
    shifted_x = x - max_x
    s = np.sum(np.exp(shifted_x), axis=axis, keepdims=True)
    # 移位公式避免概率下溢后取 log，也减少先恢复完整 LSE 再相减的精度损失。
    return shifted_x - np.log(s)
    
def softmax_backward(grad_output, probabilities, axis = -1):
    """根据前向概率和上游梯度计算 Softmax 的输入梯度。

    Args:
        grad_output: 对概率的上游梯度，与 probabilities 同形。
        probabilities: 同一输入、同一 axis 的稳定 Softmax 前向输出。
        axis: 与前向一致的归一化轴，默认为最后一轴。

    Returns:
        与 probabilities 同形的 float64 输入梯度数组。
    """
    grad_output = np.asarray(grad_output, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    # dx = p * (g - sum(g * p))；求和保留维度，以便在组内广播。
    weighted_grad = grad_output * probabilities
    correction = np.sum(weighted_grad, axis=axis, keepdims=True)
    grad_input = probabilities * (grad_output - correction)
    
    return grad_input

def logsumexp_backward(grad_output, probabilities, axis = -1):
    """将 LogSumExp 的上游梯度传播到输入。

    Args:
        grad_output: 与保留归约维度的 LSE 输出同形的上游梯度，例如 (B, 1)。
        probabilities: 同一输入、同一归约轴的稳定 Softmax 概率，与原输入同形。
        axis: 为保持反向接口一致而保留，此处不直接参与计算。

    Returns:
        与 probabilities 同形的 float64 输入梯度数组。
    """
    grad_output = np.asarray(grad_output, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    # LSE 的局部导数为 p，上游梯度按保留的归约维度广播：dx = h * p。
    grad_input = grad_output * probabilities
    return grad_input

def logsoftmax_backward(grad_output, log_probabilities, axis = -1):
    """根据前向对数概率和上游梯度计算 LogSoftmax 的输入梯度。

    Args:
        grad_output: 对对数概率的上游梯度，与 log_probabilities 同形。
        log_probabilities: 同一输入、同一 axis 的稳定 LogSoftmax 前向输出。
        axis: 与前向一致的归一化轴，默认为最后一轴。

    Returns:
        与 log_probabilities 同形的 float64 输入梯度数组。
    """
    grad_output = np.asarray(grad_output, dtype=np.float64)
    log_probabilities = np.asarray(log_probabilities, dtype=np.float64)
    # 从稳定前向的对数概率恢复 p，无需对原始分数直接取指数。
    probabilities = np.exp(log_probabilities)
    # dx = g - p * sum(g)；这里求 sum(g)，Softmax 反向求的是 sum(g * p)。
    total_grad = np.sum(grad_output, axis=axis, keepdims=True)
    grad_input = grad_output - probabilities * total_grad
    return grad_input

# 验证部分

# scores = [[1000.0, 1001.0, 1002.0],
#           [1.0, 2.0, 3.0]]
# result_softmax = softmax(scores)
# result_lse = logsumexp(scores)
# result_logsoftmax = logsoftmax(scores)

# test_grad = [[1, 0, 0], [1, 0, 0]]
# result_backward = softmax_backward(test_grad, result_softmax)
# print("梯度反传:", result_backward)
# print("梯度形状:", result_backward.shape)


# print("softmax:", result_softmax)
# print("输出形状：", result_softmax.shape)
# print("每行的概率之和：", np.sum(result_softmax, axis=1))

# print("logsumexp:", result_lse)
# print("输出形状：", result_lse.shape)

# print("logsoftmax:", result_logsoftmax)
# print("输出形状：", result_logsoftmax.shape)

# 验证部分
# p = softmax([1, 2, 3])
# assert np.allclose(p, [0.09003057, 0.24472847, 0.66524096])
# assert np.allclose(np.sum(p), 1.0)
# assert np.all(p >= 0)
# assert np.all(np.isfinite(p))

# assert np.allclose(softmax([1001, 1002, 1003]), p)
# assert np.allclose(softmax([-999, -998, -997]), p)

# assert np.allclose(softmax([5, 5, 5]), [1/3, 1/3, 1/3])

# x = np.array([
#     [1, 2, 3],
#     [1, 2, 3],
# ])

# p_rows = softmax(x, axis=1)
# assert p_rows.shape == x.shape
# assert np.allclose(np.sum(p_rows, axis=1), 1.0)

# p_columns = softmax(x, axis=0)
# assert p_columns.shape == x.shape
# assert np.allclose(p_columns, 0.5)
