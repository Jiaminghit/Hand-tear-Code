# A01：Softmax、LogSumExp 与 LogSoftmax 公式

设 $x=[x_1,\ldots,x_n]$，$m=\max_j x_j$，$\log$ 为自然对数。

## Softmax

基本公式：

$$
\operatorname{Softmax}(x)_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

数值稳定公式：

$$
\boxed{\operatorname{Softmax}(x)_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}}
$$

### 反向梯度推导

设 $p=\operatorname{Softmax}(x)$，$\mathcal L$ 为标量损失，上游梯度为 $g_i=\partial\mathcal L/\partial p_i$。记 $Z=\sum_k e^{x_k}$，$\delta_{ij}$ 在 $i=j$ 时为 1，否则为 0。

由商法则：

$$
\begin{aligned}
\frac{\partial p_i}{\partial x_j}
&=\frac{\delta_{ij}e^{x_i}Z-e^{x_i}e^{x_j}}{Z^2}\\
&=p_i(\delta_{ij}-p_j)
\end{aligned}
$$

由链式法则：

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial x_j}
&=\sum_i g_i\frac{\partial p_i}{\partial x_j}\\
&=\sum_i g_i p_i(\delta_{ij}-p_j)\\
&=\boxed{p_j\left(g_j-\sum_i g_i p_i\right)}
\end{aligned}
$$

向量化公式：

$$
\boxed{\nabla_x\mathcal L
=p\odot\left[g-\operatorname{sum}_{\mathrm{axis}}(g\odot p)\right]}
$$

其中 $\odot$ 表示逐元素相乘；求和沿前向相同的 `axis` 进行，并保留维度（`keepdims=True`）。$p$ 使用数值稳定前向得到的概率。

## LogSumExp

基本公式：

$$
\operatorname{LogSumExp}(x)=\log\sum_j e^{x_j}
$$

数值稳定公式：

$$
\boxed{\operatorname{LogSumExp}(x)=m+\log\sum_j e^{x_j-m}}
$$

### 反向梯度推导

设 $r=\operatorname{LogSumExp}(x)$，上游梯度为 $h=\partial\mathcal L/\partial r$，$p=\operatorname{Softmax}(x)$。

局部导数：

$$
\frac{\partial r}{\partial x_j}
=\frac{1}{\sum_i e^{x_i}}\,e^{x_j}
=p_j
$$

由链式法则：

$$
\frac{\partial\mathcal L}{\partial x_j}
=\frac{\partial\mathcal L}{\partial r}\frac{\partial r}{\partial x_j}
=h\,p_j
$$

向量化公式：

$$
\boxed{\nabla_x\mathcal L=h\odot p}
$$

多维输入中，$h$ 与保留归约维度的 LSE 输出同形，通过广播与 $p$ 相乘；$p$ 使用数值稳定前向得到的概率。

## LogSoftmax

基本公式：

$$
\operatorname{LogSoftmax}(x)_i
=\log\operatorname{Softmax}(x)_i
=x_i-\operatorname{LogSumExp}(x)
$$

数值稳定公式：

$$
\boxed{\operatorname{LogSoftmax}(x)_i=(x_i-m)-\log\sum_j e^{x_j-m}}
$$

### 反向梯度推导

设 $\ell_i=x_i-r$，$r=\operatorname{LogSumExp}(x)$，上游梯度为 $g_i=\partial\mathcal L/\partial\ell_i$，$p=\operatorname{Softmax}(x)$。

局部导数：

$$
\frac{\partial\ell_i}{\partial x_j}
=\delta_{ij}-\frac{\partial r}{\partial x_j}
=\delta_{ij}-p_j
$$

由链式法则：

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial x_j}
&=\sum_i g_i\frac{\partial\ell_i}{\partial x_j}\\
&=\sum_i g_i(\delta_{ij}-p_j)\\
&=\boxed{g_j-p_j\sum_i g_i}
\end{aligned}
$$

向量化公式：

$$
\boxed{\nabla_x\mathcal L
=g-p\odot\operatorname{sum}_{\mathrm{axis}}(g)}
$$

其中 $p_i=e^{\ell_i}$，由稳定 LogSoftmax 输出恢复；求和沿前向相同的 `axis` 进行，并保留维度（`keepdims=True`）。
