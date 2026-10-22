# A03：多分类交叉熵 Cross Entropy 公式

设单样本 logits 为有限实数向量 $z\in\mathbb R^C$，整数标签 $y\in\{0,\ldots,C-1\}$，$p=\operatorname{Softmax}(z)$。$\log$ 为自然对数，$\mathbf1[\cdot]$ 为指示函数。

## logits 输入的多分类交叉熵

取真实类别的负对数概率：

$$
\begin{aligned}
\ell(z,y)
&=-\log p_y\\
&=-\operatorname{LogSoftmax}(z)_y\\
&=\operatorname{LogSumExp}(z)-z_y
\end{aligned}
$$

令 $m=\max_j z_j$，数值稳定公式为：

$$
\boxed{\ell(z,y)=\log\sum_j e^{z_j-m}-(z_y-m)}
$$

直接使用稳定 LogSoftmax，避免先计算 Softmax 时概率下溢为 $0$，再取对数得到无穷值。

### 反向梯度推导

由 A01 的 LogSumExp 导数：

$$
\begin{aligned}
\frac{\partial\ell}{\partial z_j}
&=\frac{\partial\operatorname{LogSumExp}(z)}{\partial z_j}
-\frac{\partial z_y}{\partial z_j}\\
&=\boxed{p_j-\mathbf1[j=y]}
\end{aligned}
$$

即 $\nabla_z\ell=p-\operatorname{one\_hot}(y)$：真实类别位置减 $1$，其他位置保留概率值。

## 标签平滑 Label Smoothing

设平滑系数 $\varepsilon\in[0,1]$，将 one-hot 目标与均匀分布混合：

$$
\boxed{q_j=(1-\varepsilon)\mathbf1[j=y]+\frac{\varepsilon}{C}}
\qquad\sum_j q_j=1
$$

真实类别的目标概率为 $1-\varepsilon+\varepsilon/C$，其他类别为 $\varepsilon/C$。$\varepsilon=0$ 退化为硬标签；$\varepsilon=1$ 为均匀目标。

前向公式：

$$
\begin{aligned}
\ell(z,q)
&=-\sum_j q_j\log p_j\\
&=-(1-\varepsilon)\log p_y
-\frac{\varepsilon}{C}\sum_j\log p_j
\end{aligned}
$$

记 $\ell_{\mathrm{hard}}=-\log p_y$，$\ell_{\mathrm{uniform}}=-\frac1C\sum_j\log p_j$，则：

$$
\boxed{\ell=(1-\varepsilon)\ell_{\mathrm{hard}}+\varepsilon\ell_{\mathrm{uniform}}}
$$

两项均由稳定 LogSoftmax 计算。$\ell_{\mathrm{uniform}}$ 在单个样本的类别维上取平均。

### 反向梯度推导

由于 $q$ 与 logits 无关且 $\sum_jq_j=1$：

$$
\begin{aligned}
\ell
&=-\sum_jq_j\left[z_j-\operatorname{LogSumExp}(z)\right]\\
&=\operatorname{LogSumExp}(z)-\sum_jq_jz_j
\end{aligned}
$$

因此：

$$
\boxed{\frac{\partial\ell}{\partial z_j}
=p_j-q_j
=p_j-\frac{\varepsilon}{C}-(1-\varepsilon)\mathbf1[j=y]}
$$

反向直接使用稳定 Softmax 概率：所有位置减 $\varepsilon/C$，真实类别位置再减 $1-\varepsilon$。

## ignore mask、reduction 与上游梯度

batch 输入 $z$ 的形状为 $(N,C)$，整数标签 $y$ 的形状为 $(N,)$，$N,C\ge1$。Softmax 沿类别维计算。有效样本集合及数量为：

$$
V=\{i:y_i\ne\texttt{ignore\_index}\},\qquad M=|V|
$$

对有效行按前述公式计算 $q_i$ 和 $\ell_i$；被忽略行的损失与整行 logits 梯度均为零：

$$
\widetilde\ell_i=
\begin{cases}
\ell_i,&i\in V\\
0,&i\notin V
\end{cases}
\qquad
D_{i,j}=\frac{\partial\widetilde\ell_i}{\partial z_{i,j}}=
\begin{cases}
p_{i,j}-q_{i,j},&i\in V\\
0,&i\notin V
\end{cases}
$$

设最终标量损失为 $\mathcal L$，对交叉熵前向返回结果的上游梯度为 $g$：

| reduction | 前向返回结果 | logits 梯度 $\partial\mathcal L/\partial z_{i,j}$ |
|---|---|---|
| `none` | $(\widetilde\ell_i)_{i=0}^{N-1}$，形状 $(N,)$ | $g_iD_{i,j}$ |
| `sum` | $\sum_i\widetilde\ell_i$，标量 | $gD_{i,j}$ |
| `mean` | $\frac1M\sum_i\widetilde\ell_i$，标量，要求 $M>0$ | $\frac{g}{M}D_{i,j}$ |

- `none` 的上游梯度由 $(N,)$ 扩为 $(N,1)$，逐行广播；`sum`、`mean` 的上游梯度为标量。反向返回形状始终为 $(N,C)$。
- `mean` 除以有效样本数 $M$；没有忽略样本时 $M=N$。类别维的 $C$ 仅用于平滑目标中的均匀平均。
- 先筛选有效行，再按标签索引；`ignore_index` 不参与类别索引。忽略的是整个样本，标签平滑的两项损失及其梯度都需归零。
- 本练习约定：全部样本被忽略时，`none` 返回全零损失数组，`sum` 返回 $0$，两者反向均返回全零 $(N,C)$ 数组；`mean` 的前向与反向均抛出 `ValueError`。

自检性质：$\varepsilon=0$ 退化为普通交叉熵；每个样本的所有 logits 同加常数不改变损失；每行 logits 梯度之和应接近 $0$。
