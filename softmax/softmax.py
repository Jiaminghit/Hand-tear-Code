import numpy as np

def softmax(x, axis = -1):
    x = np.asarray(x, dtype=np.float64)

    max_x = np.max(x, axis = axis, keepdims=True)

    shifted_x = x - max_x

    exp_x = np.exp(shifted_x)

    possibilities = exp_x / np.sum(exp_x, axis = axis, keepdims=True)

    return possibilities

scores = [[1000.0, 1001.0, 1002.0],
          [1.0, 2.0, 3.0]]
result = softmax(scores)

print(result)
print("输出形状：", result.shape)
print("每行的概率之和：", np.sum(result, axis=1))

# 验证部分
p = softmax([1, 2, 3])
assert np.allclose(p, [0.09003057, 0.24472847, 0.66524096])
assert np.allclose(np.sum(p), 1.0)
assert np.all(p >= 0)
assert np.all(np.isfinite(p))

assert np.allclose(softmax([1001, 1002, 1003]), p)
assert np.allclose(softmax([-999, -998, -997]), p)

assert np.allclose(softmax([5, 5, 5]), [1/3, 1/3, 1/3])

x = np.array([
    [1, 2, 3],
    [1, 2, 3],
])

p_rows = softmax(x, axis=1)
assert p_rows.shape == x.shape
assert np.allclose(np.sum(p_rows, axis=1), 1.0)

p_columns = softmax(x, axis=0)
assert p_columns.shape == x.shape
assert np.allclose(p_columns, 0.5)
