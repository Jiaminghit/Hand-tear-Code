# Hand-tear-Code：AI 算法与三维重建手撕题库

记录从理论理解、独立实现到验证与限时练习的学习过程。Practice makes perfect.

本仓库面向 AI 算法与 AI 开发岗位，重点准备三维重建算法、神经渲染与仿真环境重建相关面试。题库沿四条主线展开：**通用 AI 算子、相机与三维几何、NeRF／3DGS、训练与推理工程**。

本 README 是后续学习、手撕、练习和补充仓库的总指引。当前规划包含 **A–K 共 11 个主题、139 组母题**，每组再逐步展开基础版本与变体。目录中的题目代表学习计划，不代表都已实现或验证通过。

资料整理日期：2026-10-14。

## 导航

- [Hand-tear-Code：AI 算法与三维重建手撕题库](#hand-tear-codeai-算法与三维重建手撕题库)
  - [导航](#导航)
  - [四条主线与题库范围](#四条主线与题库范围)
    - [纳入与排除](#纳入与排除)
  - [统一标记与实现边界](#统一标记与实现边界)
    - [不同题型的实现要求](#不同题型的实现要求)
  - [现有内容与学习状态](#现有内容与学习状态)
  - [A：数值计算与经典机器学习](#a数值计算与经典机器学习)
  - [B：神经网络基本模块与反向传播](#b神经网络基本模块与反向传播)
  - [C：Attention／Transformer](#cattentiontransformer)
  - [D：二维视觉与采样算子](#d二维视觉与采样算子)
  - [E：相机、多视图几何与位姿优化](#e相机多视图几何与位姿优化)
  - [F：点云、深度融合与几何评估](#f点云深度融合与几何评估)
  - [G：NeRF 与隐式神经场](#gnerf-与隐式神经场)
  - [H：3D Gaussian Splatting](#h3d-gaussian-splatting)
  - [I：仿真环境重建专项](#i仿真环境重建专项)
  - [J：AI 开发、训练工程与 GPU 算子](#jai-开发训练工程与-gpu-算子)
  - [K：其他 AI 方向选学](#k其他-ai-方向选学)
  - [近期学习顺序](#近期学习顺序)
    - [每次练习的执行顺序](#每次练习的执行顺序)
    - [后续分支](#后续分支)
  - [每道题的五部分](#每道题的五部分)
    - [建议的题目页格式](#建议的题目页格式)
    - [验证时优先检查](#验证时优先检查)
  - [参考资料与出题证据](#参考资料与出题证据)
    - [岗位范围依据](#岗位范围依据)
    - [本人面经中的代码题记录](#本人面经中的代码题记录)
    - [官方课程、文档与作者实现](#官方课程文档与作者实现)

<a id="tracks"></a>

## 四条主线与题库范围

| 主线 | 主要主题 | 学习目标 |
|---|---|---|
| **通用 AI 算子** | A、B、C、D；按方向选学 K | 理解数值稳定、损失与梯度、神经网络模块、Attention、视觉采样与后处理，能独立写出基础版本 |
| **相机与三维几何** | E、F、I 中的几何与多传感器内容 | 掌握坐标约定、相机投影、位姿、几何估计、优化、点云配准和深度融合 |
| **NeRF／3DGS** | G、H；结合 I 的动态场景内容 | 拆解射线、编码、采样、体渲染、Gaussian 投影与合成，逐步连接为小型重建或渲染项目 |
| **训练与推理工程** | J；结合 B 的优化器、C 的推理模块 | 写出可靠训练与推理流程，掌握张量操作、自定义梯度、分块处理和 GPU 算子实现 |

主题之间存在依赖与交叉，以上归类用于安排学习，不要求每道题只属于一条主线。

### 纳入与排除

- 纳入 AI 算法、模型模块、领域数值计算，以及与训练、推理、重建相关的工程编程题。
- 相机几何、ICP、RANSAC 等虽然不一定使用神经网络，但与三维重建岗位直接相关，因此纳入。
- 不在本轮目录中展开链表、排序、普通动态规划等基础数据结构与算法题；已有 [dp/](dp/) 保留为历史练习内容。
- 仿真方向以自动驾驶、机器人场景重建为主要背景。物理仿真、控制、定位融合等内容按具体岗位要求追加。

公开面经无法穷尽所有面试官的自拟题。本目录综合候选人本人面经、官方岗位要求、官方课程与作者实现，组织成可持续扩充的母题体系。**优先级是面向目标岗位的学习建议，不是出现频率统计，也不表示必考。**

<a id="markers"></a>

## 统一标记与实现边界

| 标记 | 含义 | 使用方式 |
|---|---|---|
| **P0** | 当前三维重建／仿真环境重建目标下优先掌握 | 先理解理论，再独立实现基础版本并验证 |
| **P1** | 核心题完成后继续补齐的扩展或专项 | 按依赖关系、具体 JD 和项目经历安排 |
| **P2** | 按岗位方向选择的进阶或其他方向题 | 转投相应方向时可提高优先级 |
| **★** | 本人面经明确记载过的代码题基础版本 | 只修饰标记紧邻的题目，不覆盖同一行所有变体；来源见 S1–S5 |
| **★（伪代码）** | 本人面经明确记载的思路／伪代码题 | 区别于现场完整实现；本目录 G08 对应 S6 |
| **项目／伪代码** | 完整任务较大，适合项目练习或流程说明 | 先练可独立实现的最小模块，再扩展系统；不默认要求现场完整复刻 |

未标 ★ 的题目，是根据岗位需求、官方定义或实现拆出的准备题，**不据此声称已被现场手撕考过**。候选人自述出题记录也不等于公司官方题库。理论问答、实现追问与现场编码分别看待。

### 不同题型的实现要求

- **算子与反传题**：优先使用 NumPy 或 PyTorch 基础运算实现核心公式；调用目标算子只能用于验证。例如手写 LayerNorm 时，不用现成 LayerNorm 代替核心实现。
- **模型结构题**：可使用 Linear、Embedding 等基础层，但目标模块应自行组装。例如 MHA 的拆头、Attention、合头与输出投影需要自己写。
- **几何题**：允许调用 SVD、特征分解、线性方程求解；自行构造算法方程、约束、更新与退化处理。现成 PnP、ICP 等接口用于结果对比。
- **工程题**：允许使用训练、自动求导、数据加载等框架 API，重点验证流程、状态和计算语义。
- **GPU 题**：先建立正确的 CPU／Torch 参考版本，再迁移到 CUDA，并检查数值、边界与适用时的梯度一致性。

<a id="existing"></a>

## 现有内容与学习状态

现有文件作为学习起点，统一标为 **已有初稿，待校验**。存在文件不等于已经独立掌握或通过验证；本次题库梳理不为既有实现背书。

| 已有内容 | 文件 | 对应题目 | 后续工作 |
|---|---|---|---|
| Self-Attention | [self_attention.py](attention/self_attention.py) | C01 | 核对数值稳定、mask、输入输出形状和梯度 |
| Multi-Head Self-Attention | [multihead_self_attention.py](attention/multihead_self_attention.py) | C02、C03 | 校验拆头／合头、输出投影、padding／causal mask |
| Decoder 初稿 | [decoder.py](attention/decoder.py) | C04 | 校验维度、模块调用、返回值、残差与归一化；区分不同 Decoder 结构 |
| Transformer 笔记 | [Transformer.md](attention/Transformer.md) | C 类 | 随实现校正概念、公式、维度和架构描述 |
| IoU／NMS 初稿与笔记 | [nms.py](nms/nms.py)、[notes.md](nms/notes.md) | D01、D02 | 统一 score 与坐标约定，补空输入、原索引、退化框等验证 |

后续每道题维护以下状态，按证据更新，不直接从“写过代码”跳到“完成”：

| 状态 | 达成标准 |
|---|---|
| 计划中 | 已明确题目范围、优先级和依赖 |
| 理论已学 | 能解释公式、输入输出、假设与关键边界 |
| 独立写出 | 不参考完整答案，能够写出基础实现 |
| 验证通过 | 正常与边界样例通过；适用时完成参考对齐与梯度检查 |
| 限时写出 | 能在自定面试练习时限内实现，并解释复杂度、边界和变体 |

<a id="topic-a"></a>

## A：数值计算与经典机器学习

重点是算法理解、向量化、数值稳定和边界处理。经典 ML 的较低优先级题可在目标 JD 需要时提前。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| A01 | ★ Softmax（[S1]）；LogSoftmax；LogSumExp | 任意轴、batch、稳定前向；扩展反向梯度 | P0 |
| A02 | Sigmoid；★ 二分类交叉熵 BCE（[S2]） | 分清概率输入与 logits 输入，写稳定计算和梯度 | P0 |
| A03 | ★ 多分类交叉熵（[S1]） | logits＋整数标签；扩展 label smoothing、ignore mask | P0 |
| A04 | MSE、MAE、Huber／SmoothL1 | 前向、梯度、不同 reduction、有效像素 mask | P0 |
| A05 | 线性回归、逻辑回归；★ 梯度下降（[S2]） | 损失、梯度、参数更新；扩展 L1／L2 正则 | P1 |
| A06 | ★ K-Means（[S4]）；K-Means++ | 分配、中心更新、收敛、空簇；扩展初始化 | P1 |
| A07 | PCA | 中心化、协方差、特征分解／SVD、降维与重建 | P1 |
| A08 | GMM＋EM | E-step、M-step、稳定责任度计算 | P2 |
| A09 | 高斯／多项式朴素贝叶斯 | 训练统计量、对数概率预测、平滑 | P2 |
| A10 | 决策树最佳划分 | 实现一次基于 Gini／信息增益的划分；完整树作为项目扩展 | P2 |
| A11 | KNN 分类／回归 | 距离、邻居选择、投票／加权；无需实现树索引 | P2 |
| A12 | ★ AUC（[S2]）；Precision／Recall／F1；混淆矩阵 | AUC 处理同分、全正／全负；指标明确平均方式 | P1 |

<a id="topic-b"></a>

## B：神经网络基本模块与反向传播

先用基础运算写清楚，再与框架结果对比。前向正确与反向正确分别验证。[T1] 提供了官方课程练习参考。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| B01 | Linear、激活函数、两层 MLP | PyTorch 基础运算手写前向和反向，得到输入／权重／偏置梯度；补一个 NumPy 版本，建立迁移能力 | P0 |
| B02 | ★ Conv2D（[S3]） | 基础前向；扩展 stride、padding、dilation、反向 | P0 |
| B03 | 分组卷积、Depthwise Conv、im2col | 从基础卷积扩展；核对维度和计算量 | P1 |
| B04 | MaxPool；★ AvgPool（[S3]） | 前向；扩展反向、重叠窗口梯度累加 | P1 |
| B05 | BatchNorm | 训练／推理两种模式、running statistics、仿射参数 | P0 |
| B06 | ★ LayerNorm（[S3]） | 指定归一化维度、gamma／beta、稳定方差计算；可再做一次 NumPy 迁移 | P0 |
| B07 | RMSNorm、GroupNorm、InstanceNorm | 明确统计维度与广播关系 | P1 |
| B08 | Dropout | inverted dropout；区分训练和推理 | P1 |
| B09 | SGD、Momentum、Adam、AdamW | 手写更新公式、状态维护、偏差修正 | P0 |
| B10 | Warmup、Cosine 学习率 | 给定 step 返回学习率，处理边界 | P1 |
| B11 | ★ 简化计算图反传（[S1]）；自动求导引擎 | 最小版支持加／乘、局部导数、共享节点梯度累加；完整引擎为项目 | P1 |
| B12 | Embedding 查表及反向 | 重复索引的梯度必须累加 | P1 |
| B13 | ★ InfoNCE（[S2]）；Contrastive／Triplet／CLIP Loss | 相似度矩阵、正负样本、温度、mask | P1 |
| B14 | Focal／Dice Loss；KL 蒸馏损失 | 写最小版本，明确概率／logits、温度与归一化 | P1 |

<a id="topic-c"></a>

## C：Attention／Transformer

分别练习模块与推理过程。三维方向先掌握 C01–C06 和 C11，其余结合大模型训练、推理或生成式重建需求调整。[T2]、[T3] 可作为实现参考。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| C01 | Scaled Dot-Product Attention | QKᵀ、缩放、mask、Softmax、加权 V | P0 |
| C02 | ★ Multi-Head Attention（[S1]） | 拆头、合头、输出投影，支持 batch | P0 |
| C03 | Causal／Padding／Cross-Attention | 分清 mask 含义；支持 Q 与 K 长度不同 | P0 |
| C04 | Encoder／Decoder Block | 残差、Norm、FFN；区分 Decoder-only 与 Encoder–Decoder | P1 |
| C05 | 正弦位置编码 | 批量生成、奇偶维度、输出形状 | P0 |
| C06 | RoPE | 成对旋转、位置偏移、缓存位置 | P1 |
| C07 | MQA／GQA | Q 头与 KV 头分组、广播与形状验证 | P1 |
| C08 | KV Cache | 增量解码、追加／预分配、正确 mask | P1 |
| C09 | Greedy、Temperature、Top-k／Top-p 采样 | 从 logits 得到下一个 token | P1 |
| C10 | ★ Beam Search（[S1]） | 累加 log probability、EOS、长度处理；扩展 cache 重排 | P2 |
| C11 | ViT Patchify／Unpatchify、位置 embedding | 图像与 token 的维度转换；扩展最小 ViT Block | P1 |
| C12 | LoRA Linear | 低秩分支、冻结参数、scale、merge／unmerge | P2 |
| C13 | SwiGLU；简化 MoE 路由 | 门控 FFN；MoE 写选专家、分发和合并 | P2 |
| C14 | Online Softmax／分块 Attention | 小型等价前向；完整 FlashAttention 内核作为项目／伪代码 | P2 |

<a id="topic-d"></a>

## D：二维视觉与采样算子

重点补齐插值、重映射、检测后处理和图像指标。采样算子需要明确坐标约定，评估指标需要明确计算规则。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| D01 | ★ IoU（[S3]）；GIoU／DIoU／CIoU | 单框／批量框，xyxy／xywh，空交集和退化框 | P0 |
| D02 | ★ NMS（[S3]、[S5]）；class-wise NMS；Soft-NMS | boxes＋scores 输入，返回原索引；处理空输入 | P0 |
| D03 | AP／mAP；分割 IoU／mIoU | 明确匹配、阈值、插值和类别平均规则 | P1 |
| D04 | ★ 双线性插值（[S3]） | 四邻点权重、边界策略；扩展采样位置梯度 | P0 |
| D05 | Resize／Grid Sample／图像 Warp | 反向映射、像素中心、归一化坐标与有效 mask | P0 |
| D06 | 高斯滤波、Sobel、可分离卷积 | 核构造、边界、通道处理 | P1 |
| D07 | NCC／ZNCC、Patch Matching | 局部窗口匹配与归一化 | P1 |
| D08 | 特征描述子匹配 | 最近／次近匹配、ratio test、mutual check | P1 |
| D09 | Lucas–Kanade 光流最小版 | 单点／小 patch 的一次迭代；金字塔作为扩展 | P1 |
| D10 | PSNR／SSIM | 动态范围、窗口统计、相同图像等边界 | P0 |
| D11 | 深度评估与损失 | masked L1、AbsRel、RMSE；扩展尺度不变损失 | P0 |
| D12 | ROIAlign 最小版 | bin 采样与双线性插值 | P2 |

<a id="topic-e"></a>

## E：相机、多视图几何与位姿优化

这是三维重建的重点主线。以下主要是依据岗位需求拆出的准备题。定义与算法参考见 [T4]、[T5]；允许调用 SVD 等基础数值工具。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| E01 | 针孔相机投影 | 世界点→相机点→像素；处理 Z≤0 和越界 | P0 |
| E02 | 逆投影、深度图转点云 | 区分 Z-depth 与沿射线距离，处理无效深度 | P0 |
| E03 | 坐标变换、位姿组合／求逆 | 明确 T_cw／T_wc、行／列向量、相机中心 | P0 |
| E04 | 旋转矩阵／四元数／轴角转换 | 归一化、分量顺序、小角度与特殊角度 | P0 |
| E05 | Rodrigues、SO(3)／SE(3) 指数映射 | 小角稳定分支；明确左／右扰动 | P1 |
| E06 | 重投影残差与 Jacobian | 单点到批量；用有限差分检查 | P0 |
| E07 | DLT 三角化 | 双目／多视图、齐次除法、正深度与退化 | P0 |
| E08 | 单应矩阵 DLT | 点归一化、非退化采样、反归一化 | P1 |
| E09 | 归一化八点法求 F | 构造方程、SVD、秩 2 约束 | P1 |
| E10 | F／E 转换、E 分解 | 候选 R／t、正深度选解、极线／Sampson 误差 | P1 |
| E11 | RANSAC | 先直线／平面，再几何模型；内点、退化与重估 | P0 |
| E12 | Gauss–Newton／LM | 曲线拟合或小型重投影优化，写完整迭代 | P0 |
| E13 | PnP 最小练习 | 已知初值的位姿优化；完整 EPnP／P3P 放进阶 | P1 |
| E14 | 简化 BA | 少量相机和点、固定 gauge、一次／多次更新；完整稀疏 BA 为项目／伪代码 | P1 |
| E15 | 相机畸变／去畸变 | 径向和切向模型；逆解用迭代 | P1 |
| E16 | 相机标定小模块 | 平面标定的单应约束／重投影优化；完整标定作为项目 | P2 |

<a id="topic-f"></a>

## F：点云、深度融合与几何评估

从距离、采样与已知对应配准出发，再连接 ICP、融合和学习模块。[T6]、[T7] 提供对应算法与官方实现参考。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| F01 | 点集两两距离、KNN／Ball Query | batch、排除自身、空邻域；扩展分块计算 | P0 |
| F02 | Farthest Point Sampling | 维护到已选点集的最小距离 | P1 |
| F03 | Voxel Downsampling／Voxelization | floor 索引、负坐标、同体素聚合 | P0 |
| F04 | PCA 法向与局部协方差 | 邻域、最小特征向量、符号与退化处理 | P1 |
| F05 | Kabsch／Umeyama | 已知对应点求 SE(3)／Sim(3)，处理反射 | P0 |
| F06 | Point-to-Point ICP | 对应搜索、外点剔除、配准更新与停止条件 | P0 |
| F07 | Point-to-Plane ICP | 已知法向量的残差、Jacobian 和更新 | P1 |
| F08 | Chamfer Distance；三维 F-score | 单向／双向、平方距离、阈值与点数差异 | P0 |
| F09 | ATE／RPE | 轨迹对齐、尺度规则、平移／旋转误差 | P1 |
| F10 | 视差转深度、RGB-D 重投影 | 内参、基线、单位、无效值和遮挡 | P0 |
| F11 | TSDF 融合 | 单体素／小网格的截断和加权更新；完整融合为项目 | P1 |
| F12 | MVS Cost Volume／Soft Argmin | 深度假设、warping、代价聚合、深度回归 | P1 |
| F13 | PointNet／PointNet++ 核心模块 | 共享 MLP、对称池化、采样、分组 | P1 |
| F14 | 网格法向／采样、Laplacian 平滑 | 局部计算、面积加权、边界处理 | P2 |

前馈重建方向可追加“深度／点图转点云、置信度过滤、尺度和坐标对齐”等练习。VGGT 等完整网络复现作为项目，参考 [T8]；后续追加题使用新编号，避免改变已有题目的编号含义。

<a id="topic-g"></a>

## G：NeRF 与隐式神经场

按函数拆开学习，再连接为小型神经渲染项目。NeRF 作者代码中的射线、编码、采样和体渲染函数可作为练习边界，参考 [T9]。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| G01 | 像素生成射线 | origins／directions，坐标轴、像素中心；明确方向是否归一化，并与采样距离约定一致 | P0 |
| G02 | Fourier Positional Encoding | 多频 sin／cos、输出维度、是否包含输入 | P0 |
| G03 | 射线上均匀／分层扰动采样 | near／far、区间、排序与随机性 | P0 |
| G04 | NeRF 离散体渲染 | density→alpha→透射率→权重→RGB／深度／背景 | P0 |
| G05 | PDF／CDF 重要性采样 | inverse-CDF、零权重、平坦区间、确定性版本 | P0 |
| G06 | 简化 NeRF MLP | 位置／方向分支、skip connection、RGB／density 输出 | P1 |
| G07 | 三线性插值、Hash-grid 查询 | 8 角点、哈希索引、多层特征拼接 | P0 |
| G08 | ★（伪代码）四线性插值（[S6]） | 面经要求思路与基本伪代码；练习再实现 4D 的 16 角点并扩展 GPU 版本 | P1 |
| G09 | Ray–AABB／场景内采样 | 求交区间、平行射线、无交集 | P1 |
| G10 | SDF 梯度、法向、Eikonal Loss | 求空间梯度、归一化、构造损失 | P1 |
| G11 | SDF 表面查询／Sphere Tracing | 小型查询与步进；误差、停止和最大迭代 | P2 |
| G12 | Mesh Extraction | 单元插值或小型示例；完整 Marching Cubes 作为项目／伪代码 | P2 |

**三维三线性插值使用 8 个角点，四维四线性插值使用 16 个角点。**S6 同时提及 Instant-NGP，但不能据此把标准三维 hash-grid 查询称为四线性插值，也不能将其伪代码题扩大为完整 CUDA 实现真题。

<a id="topic-h"></a>

## H：3D Gaussian Splatting

先掌握 Gaussian 的表示、投影与合成，再连接渲染器和优化过程。[T10]、[T11] 提供可直接对照的算子拆分与作者实现。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| H01 | 三维 Gaussian 协方差构造 | quaternion＋scale→R diag(s²) Rᵀ，保证旋转与尺度约定正确 | P0 |
| H02 | Gaussian 世界／相机变换 | 均值与协方差一起变换 | P0 |
| H03 | 三维 Gaussian 投影到二维 | 投影 Jacobian、二维协方差、稳定项 | P0 |
| H04 | 二维 Gaussian／Conic 求像素 alpha | 坐标差、二次型、opacity 与数值边界 | P0 |
| H05 | Front-to-Back Alpha Compositing | 深度顺序、透射率、背景与 early stop | P0 |
| H06 | 球谐函数 SH 颜色 | 先 degree 0／1，后续扩展更高阶 | P1 |
| H07 | 屏幕包围范围、Tile 覆盖 | 半径估计、屏幕裁剪、tile 范围 | P1 |
| H08 | Densify／Clone／Split／Prune | 条件、参数复制与筛除；先伪代码／最小实现，优化器状态迁移为进阶 | P1 |
| H09 | 小型 Gaussian Rasterizer | CPU／Torch 前向串联以上模块；项目 | P1 |
| H10 | 可微 Rasterizer Backward | 从单 Gaussian／单像素梯度开始；完整内核为项目／伪代码 | P2 |

第一阶段优先验证 H01–H05 的最小模块，再开始 H09；完整渲染器及其反向内核不当作单个普通现场题。

<a id="topic-i"></a>

## I：仿真环境重建专项

依据动态街景、多传感器、可交互资产或物理仿真需求追加。动态城市重建可参考 [T12]。I09、I10 是否提前，取决于岗位是否明确涉及定位融合或物理仿真。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| I01 | LiDAR→Camera 投影与深度图 | 外参、内参、最近正深度、有效 mask／z-buffer | P0 |
| I02 | 位姿时间插值／SLERP | 平移插值、旋转最短路径、q 与 −q | P1 |
| I03 | 点云运动补偿／去畸变 | 每点时间戳变换到统一参考时刻 | P1 |
| I04 | 动态目标 Local↔World 变换 | 时间相关位姿、点与协方差变换 | P1 |
| I05 | 多帧深度／图像一致性损失 | 重投影、采样、遮挡与有效观测 mask | P0 |
| I06 | BEV／Occupancy 聚合 | 点或体素特征 scatter、重复索引聚合 | P1 |
| I07 | Ray–Triangle、重心坐标、Z-buffer | 单三角形／小场景的求交或光栅化 | P2 |
| I08 | SDF 碰撞查询与表面法向 | 查询距离、梯度和简单穿透判断 | P2 |
| I09 | KF／EKF、IMU 积分小模块 | 预测、更新、Jacobian；定位融合岗位提高优先级 | P2 |
| I10 | Euler 积分、弹簧质点、碰撞响应 | 最小动力学步骤；物理仿真岗位选学 | P2 |

完整 FEM、MPM、物理引擎等系统不默认列入视觉场景重建核心题库；确有相关 JD 或项目需求时，再单独规划。

<a id="topic-j"></a>

## J：AI 开发、训练工程与 GPU 算子

这里包含现场算法题与工程编程题。CUDA 先建立正确参考实现，再验证内核；自定义算子集成可参考 [T13]。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| J01 | Tensor 向量化与维度操作 | broadcast、reshape、permute、gather／scatter、非连续张量 | P0 |
| J02 | 最小 Train／Eval Loop | 清梯度、反传、更新、评估、正确统计 loss | P0 |
| J03 | 梯度累积、梯度裁剪、AMP | 更新时机、缩放、unscale；验证等效 batch 的适用条件 | P0 |
| J04 | Dataset／Collate／变长 Batch | padding、mask、图像或多视图样本组织 | P1 |
| J05 | Checkpoint 保存与恢复 | 参数、优化器、scheduler、step、随机状态 | P1 |
| J06 | 自定义 autograd.Function | 手写 forward／backward，用数值梯度核对 | P1 |
| J07 | 分块推理／渲染 | 控制临时张量大小，输出拼接与数值一致 | P0 |
| J08 | ★ CUDA Reduction／Stable Softmax（[S3]） | sum／max、块内归约、边界和稳定计算 | P1 |
| J09 | ★ CUDA Transpose（[S3]）；GEMM | 转置；naive→shared-memory 分块乘法 | P1 |
| J10 | ★ GPU Pool／IoU（[S3]）；插值／Norm 内核扩展 | 先从 S3 的 CPU／C++ 插值和 Norm 题写参考版，再迁移 GPU 并对齐 | P1 |
| J11 | 对称／非对称量化、Quant／Dequant | scale、zero point、clamp、误差 | P2 |
| J12 | 分布式训练小模块 | 数据划分、loss／gradient 聚合；完整训练框架为项目／伪代码 | P2 |

S3 对 CUDA 与 C++ 题目分别列举，不能将所有 C++ 题都视作已证实 CUDA 真题。GPU 版本是合理练习扩展。部署优化、GPU 编程、AI Infra 岗位可提高 J08–J12 的优先级。

<a id="topic-k"></a>

## K：其他 AI 方向选学

本主题用于保留 AI 算法与 AI 开发的更广覆盖面。**对当前三维目标统一为 P2**；转投相应方向时再调整。K12–K13 属于 AI 应用工程，练习方式与手写神经网络算子有所不同。

| 编号 | 母题与主要变体 | 建议写到什么程度 | 优先级 |
|---|---|---|---|
| K01 | RNN／LSTM／GRU | 单步前向、序列展开；扩展 BPTT | P2 |
| K02 | VAE | 重参数化、重建损失、KL Loss | P2 |
| K03 | GAN | 最小生成器／判别器训练步骤与损失 | P2 |
| K04 | Diffusion | 加噪、noise schedule、噪声预测训练损失 | P2 |
| K05 | DDPM／DDIM、Classifier-Free Guidance | 单步采样和条件／无条件预测组合 | P2 |
| K06 | VQ-VAE／Vector Quantization | 最近码本查询、量化、commitment loss | P2 |
| K07 | Q-Learning／DQN | TD target、终止 mask、目标网络更新 | P2 |
| K08 | REINFORCE／Actor–Critic | 回报、baseline、策略损失 | P2 |
| K09 | PPO／GAE | advantage 递推、ratio、clip objective | P2 |
| K10 | 推荐算法：MF／FM／BPR；★ DIN（[S2]） | 各自最小前向或训练目标；DIN 明确历史行为 mask | P2 |
| K11 | NLP／语音：CRF／CTC 解码 | 小规模动态规划、Greedy／Beam；专项岗位选学 | P2 |
| K12 | RAG 检索模块 | 文本分块、余弦检索、RRF、检索指标；小型工程练习 | P2 |
| K13 | Agent 应用编程 | 最小工具调用循环、结构化输出、重试与流式处理；小型工程练习 | P2 |

<a id="roadmap"></a>

## 近期学习顺序

近期按依赖关系推进，不预设每周可投入的时间。以下是学习路线，不是要求把整个题库串行学完；P1／P2 根据 JD 和项目需求穿插。

**当前入口：A01「稳定 Softmax、LogSoftmax、LogSumExp」。**它能衔接交叉熵、Attention、InfoNCE 和 CUDA Softmax。已有 C01、C02、C04、D01、D02 在对应阶段穿插校验，不直接记为完成。

| 阶段 | 建议学习顺序 | 阶段目标 |
|---|---|---|
| 1：数值、梯度与训练基础 | A01→A02→A03→A04→B01→B06→B05→B09；配合 J01、J02、J03 | 能解释稳定计算与反向，写出最小训练流程；后续几何任务可先用基础 SGD，再扩展 AdamW |
| 2：Attention 与视觉采样 | C01→C02→C03→C05；D01→D02→D04→D05；已有 C04 随 B06、C03 学习校验 | 校验已有初稿；能正确处理形状、mask、采样坐标与后处理边界 |
| 3：相机、位姿与点云 | E03→E01→E02→E04→E07；A07→F05；E11；E05→E06→E12；F01→F06→F08→F03 | 建立坐标约定，写出投影、配准、鲁棒估计与最小优化；A07 提供 SVD／协方差直觉 |
| 4：NeRF／3DGS 核心 | G01→G02→G03→G04→G05→G07；H01→H02→H03→H04→H05 | 独立写出采样、体渲染、Gaussian 投影与合成；配合 D10、D11 检查输出 |
| 5：连接场景与工程 | I01、I05、J07；按需求补 I02、I04、H06–H09、J06、J08–J10 | 把小模块连接成动态场景或渲染练习；选择一个算子做 GPU 迁移与验证 |

### 每次练习的执行顺序

1. 先确定题目定义、允许调用的工具与目标实现边界。
2. 学理论、推公式，用小例子检查形状与坐标。
3. 不看完整答案，写最小可运行实现。
4. 验证正常、边界和退化情形；适用时检查参考结果与梯度。
5. 修正笔记与实现，记录错误原因，再做一次独立或限时复写。
6. 达到对应学习状态后补充题目文档与代码，再提交到仓库。

### 后续分支

- **SfM／MVS／SLAM**：优先深化 E08–E16、F07、F09–F12；定位融合相关岗位追加 I09。
- **动态街景／多传感器重建**：优先深化 I01–I06、H06–H09。
- **渲染器／GPU／部署优化**：优先深化 J06、J08–J12，并选择 H09／H10 作为项目。
- **前馈或生成式重建**：深化 C 类视觉模块、F 类点图处理；按需求选 K02、K04–K06。
- **其他 AI 岗位**：使用 K 类选学目录重新调整优先级，不要求完成所有分支后才开始投递。

<a id="problem-format"></a>

## 每道题的五部分

后续新增或完善题目，统一包含以下五部分。编号保持与本 README 一致；变体可使用子编号，例如 A01-01、A01-02，避免重复创建同一母题。

| 部分 | 应包含的内容 | 完成时应能回答的问题 |
|---|---|---|
| **1. 题目定义** | 输入、输出、形状、dtype／device、坐标或概率约定、允许调用的库、基础版本与变体边界 | 给定什么，返回什么？哪些条件必须约定？ |
| **2. 理论与推导** | 核心公式、推导过程、假设、数值稳定、适用条件；涉及梯度时说明链式关系 | 为什么这样计算？公式在哪些情况下成立？ |
| **3. 最小手撕实现** | 先清晰基础版，再按需做向量化、batch、分块或 GPU；注明使用的基础算子 | 能否独立写出并解释每一步？ |
| **4. 验证** | 正常样例、边界／退化样例、参考实现对比与误差容限；涉及反向时做梯度检查 | 如何证明实现符合定义，而不是只在一个例子上能运行？ |
| **5. 面试追问与变体** | 时间／空间复杂度、数值稳定、batch／mask、GPU、退化处理、扩展题与限制 | 需求改变后，哪些部分要改？代价和风险是什么？ |

### 建议的题目页格式

每道题的笔记或说明页可使用如下顺序，文件组织可按现有目录逐步调整：

1. 题目编号、名称、优先级、当前状态、资料来源。
2. 题目定义。
3. 理论与推导。
4. 最小手撕实现及代码入口。
5. 验证方法、样例和结果。
6. 面试追问与变体。
7. 复盘：本次错误、原因、修正与下一次复写记录。

可用“理论已学、独立写出、验证通过、限时写出”记录进度。无需为没有意义的形式检查增加负担；验证围绕真实边界、数值、梯度、坐标和行为展开。

### 验证时优先检查

- **数值题**：极大／极小输入、溢出、零分母、全同值、不同 reduction。
- **张量题**：batch、非连续张量、广播、mask、长度不同、空输入，以及适用时的 dtype／device。
- **几何题**：坐标变换方向、单位、正深度、齐次除法、退化构型、小角度与反射。
- **渲染题**：采样顺序、透射率、背景、空观测、截断与有效性。
- **反向题**：有限差分或 gradcheck、共享节点／重复索引累加、参考梯度对齐。

更新题库时保留已有编号含义。新增题目、变体、来源与完成状态同步维护，避免把计划目录误当成完成清单。

<a id="references"></a>

## 参考资料与出题证据

### 岗位范围依据

| 来源 | 对本题库的用途 |
|---|---|
| [博世：三维重建算法工程师_BCSC](https://jobs.smartrecruiters.com/boschgroup/744000133497254) | 三维几何、位姿优化、神经重建、多传感器、动态场景与仿真集成的岗位范围依据 |
| [北云：三维重建与高精建图算法工程师](https://www.bynav.com/cn/careers/campus-recruitment/recruitment/1542.html) | 点云配准、SLAM、3DGS 与大场景建图的岗位范围依据 |

岗位页面会变化。本目录记录整理时的准备范围，不保证岗位持续开放，也不将职责表述直接等同于面试手撕题。

### 本人面经中的代码题记录

| 编号 | 来源 | 可支持的范围 |
|---|---|---|
| S1 | [2022 年 NLP 算法本人面经][S1] | MHA、Decoder 推理／Beam Search、Softmax／交叉熵、简化计算图反传 |
| S2 | [2024 年搜广推本人面经][S2] | AUC、BCE、InfoNCE、梯度下降、DIN |
| S3 | [2024 年部署／推理优化／高性能计算本人面经][S3] | CUDA reduction、Softmax、矩阵转置、平均池化、bbox IoU；C++ NMS、Conv2D、双线性插值、LayerNorm |
| S4 | [阿里 CV 算法岗本人面经][S4] | 欧氏距离与 K-Means 编码；K-Means++ 是问答，不据此标为完整手撕 |
| S5 | [视觉算法实习本人面经][S5] | NMS 现场手撕记录 |
| S6 | [2024 年三维重建／SLAM 本人面经][S6] | 四线性插值的思路与基本伪代码；高斯 tile 等是实现追问，不据此称完整编码题 |

上述记录只支持特定候选人、场景与基础版本。未标 ★ 的扩展仍有学习价值，但不能扩大真实出题证据的范围。

### 官方课程、文档与作者实现

| 编号 | 资料 | 主要用途 |
|---|---|---|
| T1 | [Stanford CS231n 官方作业][T1] | 前反传、归一化、Dropout、CNN 与训练练习 |
| T2 | [Meta Llama 模型实现][T2] | Attention、RMSNorm、RoPE、KV Cache、门控 FFN |
| T3 | [Microsoft LoRA 实现][T3] | 低秩分支、冻结、scale、merge／unmerge |
| T4 | [OpenCV 相机标定与三维重建文档][T4] | 相机模型、投影、PnP、三角化与多视图几何 |
| T5 | [Stanford CS231A 课程笔记][T5] | 相机、对极几何、立体视觉与鲁棒估计 |
| T6 | [Open3D 官方 ICP 教程][T6] | 点到点／点到平面配准目标与迭代 |
| T7 | [PointNet++ 作者实现][T7] | 点云采样、分组与分层特征学习 |
| T8 | [VGGT 官方仓库][T8] | 前馈重建项目与输出处理扩展 |
| T9 | [NeRF 作者实现][T9] | 射线、编码、采样与体渲染 |
| T10 | [gsplat 官方算子文档][T10] | Gaussian 协方差、投影、SH、tile 与像素合成 |
| T11 | [3DGS 作者可微光栅化实现][T11] | Gaussian 投影与前向／反向渲染模块 |
| T12 | [Street Gaussians 作者实现][T12] | 动态城市场景与目标变换项目 |
| T13 | [PyTorch 自定义 C++／CUDA 算子教程][T13] | 算子集成与验证 |

学习时先阅读题目定义与相关理论，再使用官方实现校验思路和结果。参考代码是验证与扩展依据，不替代独立手撕。

[S1]: https://www.nowcoder.com/discuss/361611315641692160
[S2]: https://www.nowcoder.com/discuss/667864275997687808
[S3]: https://www.nowcoder.com/discuss/599177965083054080
[S4]: https://www.nowcoder.com/discuss/353156919047168000
[S5]: https://www.nowcoder.com/discuss/353154978036523008
[S6]: https://www.nowcoder.com/feed/main/detail/057f10b935f545dbad249a1a62e987e4
[T1]: https://cs231n.github.io/assignments2025/assignment2/
[T2]: https://github.com/meta-llama/llama/blob/main/llama/model.py
[T3]: https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
[T4]: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
[T5]: https://web.stanford.edu/class/cs231a/course_notes.html
[T6]: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
[T7]: https://github.com/charlesq34/pointnet2
[T8]: https://github.com/facebookresearch/vggt
[T9]: https://github.com/bmild/nerf
[T10]: https://docs.gsplat.studio/main/apis/utils.html
[T11]: https://github.com/graphdeco-inria/diff-gaussian-rasterization
[T12]: https://github.com/zju3dv/street_gaussians
[T13]: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
