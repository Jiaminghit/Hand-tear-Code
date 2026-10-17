# A02：Sigmoid 与二分类交叉熵 BCE 公式

设 $z$ 为有限实数 logit，$p=\sigma(z)$ 为正类概率，$y\in[0,1]$ 为标签（硬标签为 $0$ 或 $1$）。$\log$ 为自然对数，$\odot$ 表示逐元素相乘。

计算关系：$z\xrightarrow{\mathrm{Sigmoid}}p\xrightarrow{\mathrm{BCE}(\cdot,y)}\ell$。将两步合并，可直接由 $z$ 稳定计算损失。

## Sigmoid

基本公式：

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

与 A01 的联系：

$$
\operatorname{Softmax}([0,z])=[1-\sigma(z),\ \sigma(z)]
$$

Sigmoid 逐元素计算，不沿某个轴归一化。

数值稳定公式：

$$
\boxed{
\sigma(z)=
\begin{cases}
\dfrac{1}{1+e^{-z}},&z\ge0\\[6pt]
\dfrac{e^z}{1+e^z},&z<0
\end{cases}
}
$$

分支内指数的输入均不大于 $0$，避免指数上溢；数组实现按正负掩码分别计算。

### 反向梯度推导

$$
\frac{\partial p}{\partial z}
=\frac{e^{-z}}{(1+e^{-z})^2}
=p(1-p)
$$

设 $\mathcal L$ 为最终标量损失，上游梯度为 $g=\partial\mathcal L/\partial p$，由链式法则：

$$
\boxed{\nabla_z\mathcal L=g\odot p\odot(1-p)}
$$

反向可复用前向概率 $p$，无需重新计算指数。

## 概率输入的 BCE

对硬标签，模型赋给真实标签的概率为 $P(y\mid z)=p^y(1-p)^{1-y}$。取负对数得到：

$$
\boxed{\ell(p,y)=-y\log p-(1-y)\log(1-p)}
$$

其中 $0<p<1$。硬标签下：

$$
\ell(p,y)=
\begin{cases}
-\log p,&y=1\\
-\log(1-p),&y=0
\end{cases}
$$

真实类别的预测概率越低，损失越大；同一 BCE 公式也适用于软标签 $y\in[0,1]$。

### 反向梯度推导

$$
\begin{aligned}
\frac{\partial\ell}{\partial p}
&=-\frac{y}{p}+\frac{1-y}{1-p}\\
&=\boxed{\frac{p-y}{p(1-p)}}
\end{aligned}
$$

这是损失对概率 $p$ 的局部导数；实际反向还需结合 `reduction` 并乘以上游梯度。

## logits 输入的 BCE

即使 Sigmoid 前向没有上溢，浮点概率仍可能舍入为 $0$ 或 $1$，使后续对数或除法产生无穷值、`NaN`。因此将 Sigmoid 与 BCE 合并，直接接收原始分数 $z$。

记 $A(z)=\log(1+e^z)=\operatorname{LogSumExp}([0,z])$，则：

$$
\log p=z-A(z),\qquad\log(1-p)=-A(z)
$$

代入 BCE：

$$
\begin{aligned}
\ell(z,y)
&=-y[z-A(z)]+(1-y)A(z)\\
&=A(z)-yz\\
&=\log(1+e^z)-yz
\end{aligned}
$$

沿用 A01 的稳定 LogSumExp，令 $m=\max(z,0)$：

$$
A(z)=m+\log(e^{-m}+e^{z-m})
=\max(z,0)+\log(1+e^{-|z|})
$$

数值稳定公式：

$$
\boxed{\ell(z,y)=\bigl[\max(z,0)-yz\bigr]+\operatorname{log1p}(e^{-|z|})}
$$

其中 `log1p(t)` 更准确地计算 $\log(1+t)$。先计算方括号中的主项，再加微小修正项，避免先加后减大数造成精度损失。

### 与 Sigmoid 串联后的梯度

$$
\begin{aligned}
\frac{\partial\ell}{\partial z}
&=\frac{\partial\ell}{\partial p}\frac{\partial p}{\partial z}\\
&=\frac{p-y}{p(1-p)}\,p(1-p)\\
&=\boxed{p-y=\sigma(z)-y}
\end{aligned}
$$

实现时直接计算 $\sigma(z)-y$，避免分别计算两个因子后出现 $\mathrm{inf}\times0$；极小梯度仍可能因浮点精度而舍入为 $0$。

## reduction 与上游梯度

设共有 $M$ 个损失元素，$u$ 表示 BCE 的输入，局部导数为：

$$
d_i=\frac{\partial\ell_i}{\partial u_i}=
\begin{cases}
\dfrac{p_i-y_i}{p_i(1-p_i)},&u=p\quad\text{（概率输入）}\\[6pt]
p_i-y_i,&u=z\quad\text{（logits 输入）}
\end{cases}
$$

设最终标量损失为 $\mathcal L$，对 BCE 返回结果的上游梯度为 $g$：

| reduction | 前向返回结果 | 输入梯度 $\partial\mathcal L/\partial u_i$ |
|---|---|---|
| `none` | 每个元素的 $\ell_i$ | $g_i d_i$ |
| `sum` | $\sum_{i=1}^{M}\ell_i$ | $g d_i$ |
| `mean` | $\frac1M\sum_{i=1}^{M}\ell_i$ | $\frac{g}{M}d_i$ |

`none` 的上游梯度与损失同形；`sum`、`mean` 的上游梯度为标量。$M$ 是所有参与平均的元素数，例如形状为 $[B,C]$ 时 $M=BC$。

采用“Sigmoid → 概率版 BCE”串联反向时，`mean` 的缩放只在 BCE 反向中处理一次，再将结果作为 Sigmoid 的上游梯度；logits 版 BCE 直接返回对 $z$ 的梯度。
