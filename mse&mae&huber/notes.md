# A04：MSE、MAE、Huber 与 SmoothL1 公式

设预测值为 $\hat y$，真实值为 $y$，两者同形。记逐元素残差 $r_i=\hat y_i-y_i$，真实值与损失参数视为常量，因此 $\partial r_i/\partial\hat y_i=1$。

以下先给出逐元素损失 $\ell_i$ 与局部梯度 $d_i$，再统一处理 `reduction`、上游梯度和 mask；不可导处的 $d_i$ 表示约定的次梯度。

## MSE：平方误差

前向及梯度推导：

$$
\ell_i=r_i^2,\qquad
d_i=\frac{\partial\ell_i}{\partial\hat y_i}
=2r_i\frac{\partial r_i}{\partial\hat y_i}
=\boxed{2r_i}
$$

取平均后为 $L_{\mathrm{MSE}}=\frac1M\sum_i r_i^2$，$M$ 为全部元素数。此处不乘 $1/2$；误差越大，局部梯度的绝对值越大。

## MAE：绝对误差

前向公式：

$$
\ell_i=|r_i|=
\begin{cases}
r_i,&r_i\ge0\\
-r_i,&r_i<0
\end{cases}
$$

非零残差处分别对 $r_i$、$-r_i$ 求导，得到 $1$、$-1$。零残差处左右导数不同，普通导数不存在；从次梯度集合 $[-1,1]$ 中选择 $0$：

$$
\boxed{d_i=\operatorname{sign}(r_i)=
\begin{cases}
1,&r_i>0\\
0,&r_i=0\\
-1,&r_i<0
\end{cases}}
$$

取平均后为 $L_{\mathrm{MAE}}=\frac1M\sum_i|r_i|$。非零残差处，局部梯度只取决于符号，不随误差大小增长。

## Huber

阈值 $\delta$ 为有限正数。前向公式：

$$
\ell_i=
\begin{cases}
\frac12r_i^2,&|r_i|\le\delta\\[4pt]
\delta\left(|r_i|-\frac12\delta\right),&|r_i|>\delta
\end{cases}
$$

分别求导：二次段为 $r_i$，线性段为 $\delta\operatorname{sign}(r_i)$，所以：

$$
d_i=
\begin{cases}
r_i,&|r_i|\le\delta\\
\delta\operatorname{sign}(r_i),&|r_i|>\delta
\end{cases}
=\boxed{\operatorname{clip}(r_i,-\delta,\delta)}
$$

在 $|r_i|=\delta$ 处，两段损失均为 $\delta^2/2$，导数也相同；在 $r_i=0$ 处普通导数为 $0$。小误差段是平方误差的一半，大误差段的局部梯度绝对值为 $\delta$。

## SmoothL1

阈值 $\beta$ 为有限非负数。先考虑 $\beta>0$，前向公式为：

$$
\ell_i=
\begin{cases}
\frac{r_i^2}{2\beta},&|r_i|<\beta\\[4pt]
|r_i|-\frac12\beta,&|r_i|\ge\beta
\end{cases}
$$

分别求导：二次段为 $2r_i/(2\beta)=r_i/\beta$，线性段为 $\operatorname{sign}(r_i)$，所以：

$$
d_i=
\begin{cases}
\frac{r_i}{\beta},&|r_i|<\beta\\[4pt]
\operatorname{sign}(r_i),&|r_i|\ge\beta
\end{cases}
=\boxed{\frac{\operatorname{clip}(r_i,-\beta,\beta)}{\beta}}
$$

在 $|r_i|=\beta$ 处，两段损失均为 $\beta/2$，导数也相同。大误差段的局部梯度绝对值固定为 $1$。

**$\beta=0$ 时单独退化为 MAE**：$\ell_i=|r_i|$，$d_i=\operatorname{sign}(r_i)$，零残差处选择零次梯度。实现时先判断此分支，避免除零。

## Huber 与 SmoothL1 的区别

只有在阈值相同且为正，即 $\delta=\beta=t>0$ 时，才有以下对应关系：

$$
\boxed{\ell_{\mathrm{SmoothL1},t}=\frac{\ell_{\mathrm{Huber},t}}{t}}
\qquad
\boxed{d_{\mathrm{SmoothL1},t}=\frac{d_{\mathrm{Huber},t}}{t}}
$$

第二式由第一式对预测值求导得到，因为 $t$ 是常量。

| 对比项 | Huber | SmoothL1 |
|---|---|---|
| 小误差段局部梯度 | $r_i$ | $r_i/t$ |
| 大误差段局部梯度 | $t\operatorname{sign}(r_i)$ | $\operatorname{sign}(r_i)$ |
| 局部梯度绝对值上限 | $t$ | $1$ |

$t=1$ 时两种损失函数完全相同；$t=2$ 时 SmoothL1 的损失与局部梯度均为对应 Huber 的一半。相同 mask、reduction 和上游梯度下，最终反向结果也满足该缩放关系。$t=0$ 不适用上述除法关系。

## reduction、mask 与上游梯度

布尔 mask 与输入完全同形，`True` 表示有效；未提供 mask 时全部有效。设有效位置集合及数量为：

$$
V=\{i:\mathrm{mask}_i=\mathrm{True}\},\qquad K=|V|
$$

只在有效位置计算前述 $\ell_i$ 和 $d_i$，然后放回原形状，无效位置填零：

$$
\widetilde\ell_i=
\begin{cases}
\ell_i,&i\in V\\
0,&i\notin V
\end{cases}
\qquad
D_i=
\begin{cases}
d_i,&i\in V\\
0,&i\notin V
\end{cases}
$$

设最终标量目标为 $\mathcal L$，对前向返回结果的上游梯度为 $g$。由链式法则，局部梯度乘以上游梯度；前向取平均时，再乘 $1/K$：

| reduction | 前向返回结果 | 预测值梯度 $\partial\mathcal L/\partial\hat y_i$ |
|---|---|---|
| `none` | $\widetilde\ell$，与输入同形 | $g_iD_i$ |
| `sum` | $\sum_{i\in V}\ell_i$，标量 | $gD_i$ |
| `mean` | $\frac1K\sum_{i\in V}\ell_i$，标量，要求 $K>0$ | $\frac{g}{K}D_i$ |

- `none` 的上游梯度与输入同形；`sum`、`mean` 的上游梯度为标量。反向返回值始终与预测值同形。
- `mean` 的前向与反向都除以有效元素数 $K$，有效但误差为零的位置仍计入分母。全部有效时 $K=M$，例如输入形状为 $(B,C)$，则 $M=BC$。
- **先筛选有效位置，再计算残差、损失与梯度**。不能先算全量结果再乘 mask，因为 $\mathrm{NaN}\times0$ 仍为 `NaN`。`none` 的反向也只读取有效位置的上游梯度，无效位置直接保留零。
- 本练习要求输入非空、有效位置的预测与真值有限；mask 为同形布尔数组，不做广播。
- 全部位置无效时：`none` 返回同形全零损失，`sum` 返回 $0$，两者反向均返回同形全零数组；`mean` 的前向与反向均抛出 `ValueError`。
- Huber／SmoothL1 的梯度上限指局部梯度，乘以上游梯度后，最终返回值可能超过该上限。

实现对应：`RegressionLoss` 统一处理输入、mask、reduction 与上游梯度；各子类仅通过 `_elementwise_loss` 和 `_elementwise_grad` 提供逐元素公式。
