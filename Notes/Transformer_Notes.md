# Transformer 架构详解：Encoder-Decoder 架构与推理过程

> 本文以 **Seq2Seq Attention (Encoder-Decoder Transformer)** 为例，系统梳理 Transformer 的 Encoder/Decoder 架构、核心模块原理，并重点阐述推理（Inference）过程的完整流程。每一部分均附带**输入/输出维度示例**帮助理解。

---

## 目录

- [一、整体架构概览](#一整体架构概览)
- [二、Encoder 详解](#二encoder-详解)
  - [2.1 Encoder 的输入与输出](#21-encoder-的输入与输出)
  - [2.2 Self-Attention 机制](#22-self-attention-机制)
  - [2.3 Multi-Head Attention](#23-multi-head-attention)
  - [2.4 位置编码 Positional Encoding](#24-位置编码-positional-encoding)
  - [2.5 前馈网络 FFN](#25-前馈网络-ffn)
  - [2.6 LayerNorm & 残差连接](#26-layernorm--残差连接)
  - [2.7 Encoder 完整前向过程](#27-encoder-完整前向过程)
- [三、Decoder 详解](#三decoder-详解)
  - [3.1 Decoder 的输入与输出](#31-decoder-的输入与输出)
  - [3.2 Masked Self-Attention & Causal Mask](#32-masked-self-attention--causal-mask)
  - [3.3 Cross-Attention (Encoder-Decoder Attention)](#33-cross-attention-encoder-decoder-attention)
  - [3.4 Decoder 完整前向过程](#34-decoder-完整前向过程)
- [四、推理过程详解](#四推理过程详解)
  - [4.1 推理 vs 训练的核心区别](#41-推理-vs-训练的核心区别)
  - [4.2 Encoder 推理](#42-encoder-推理)
  - [4.3 Decoder 自回归推理](#43-decoder-自回归推理)
  - [4.4 KV Cache 优化](#44-kv-cache-优化)
  - [4.5 完整推理流程示例](#45-完整推理流程示例)
- [五、维度汇总与实例](#五维度汇总与实例)
  - [5.1 各模块维度速查表](#51-各模块维度速查表)
  - [5.2 完整数值示例：英译中翻译](#52-完整数值示例英译中翻译)

---

## 一、整体架构概览

Transformer（Vaswani et al., 2017）采用 **Encoder-Decoder** 架构：

```
┌──────────────────────────────────────────────────────────────┐
│                    Transformer (Seq2Seq)                      │
├──────────────────────────┬───────────────────────────────────┤
│       Encoder            │           Decoder                 │
│   (理解输入序列)          │      (生成输出序列)                │
├──────────────────────────┼───────────────────────────────────┤
│  Input Embedding         │  Output Embedding                 │
│  + Positional Encoding   │  + Positional Encoding           │
│         ↓                │         ↓                         │
│  ┌──────────────────┐    │  ┌──────────────────┐             │
│  │ EncoderLayer ×N   │    │  │ Masked Self-Attn │             │
│  │  - MultiHead Attn │    │  └────────┬─────────┘             │
│  │  - FFN            │    │           ↓                      │
│  │  - LayerNorm      │    │  ┌──────────────────┐             │
│  └──────────────────┘    │  │ Cross-Attention   │←── Encoder  │
│         ↓                │  │ (Q=Decoder,       │    输出 K,V │
│  输出: 上下文表示         │  │  K,V=Encoder)     │             │
│  (K, V 供 Decoder 查询)   │  └────────┬─────────┘             │
│                         │           ↓                      │
│                         │  ┌──────────────────┐             │
│                         │  │ FFN              │             │
│                         │  └────────┬─────────┘             │
│                         │           ↓                      │
│                         │  Linear + Softmax                │
│                         │  → 预测下一个 token              │
└──────────────────────────┴───────────────────────────────────┘
```

**核心思想**：

| 组件 | 作用 | 类比 |
|------|------|------|
| **Encoder** | 读取完整输入序列，生成上下文表示 | "阅读理解全文" |
| **Decoder** | 基于 Encoder 的上下文，逐个生成输出 token | "根据理解逐句翻译" |
| **Self-Attention** | 序列内任意两位置直接交互 | "全体讨论，按重要性投票" |
| **Cross-Attention** | Decoder 查询 Encoder 的输出 | "查笔记，看原文对应位置" |
| **Causal Mask** | 防止 Decoder 偷看未来 token | "写作时不能看还没写的句子" |

**超参数约定**（本文统一使用以下数值演示）：

| 符号 | 值 | 含义 |
|------|-----|------|
| \(d_{model}\) | 512 | 模型隐层维度 |
| \(h\) | 8 | 注意力头数 |
| \(d_k = d_v\) | 64 | 每头维度 (512/8=64) |
| \(d_{ff}\) | 2048 | FFN 中间层维度 (4×512) |
| \(N\) | 6 | Encoder/Decoder 层数 |
| \(T_{enc}\) | 10 | Encoder 输入序列长度（举例） |
| \(T_{dec}\) | 8 | Decoder 输出序列长度（举例） |

---

## 二、Encoder 详解

### 2.1 Encoder 的输入与输出

**输入**：一个 token 序列，每个 token 被映射为 \(d_{model}\) 维向量。

**输入维度**：

\[
X_{input} \in \mathbb{R}^{T_{enc} \times d_{model}}
\]

- \(T_{enc}\)：输入序列长度（token 个数）
- \(d_{model}\)：每个 token 的表示维度

**输出维度**：

\[
X_{enc\_out} \in \mathbb{R}^{T_{enc} \times d_{model}}
\]

- 序列长度不变，每个位置获得了**全局上下文信息**

**数值示例**：

假设输入句子为 "The cat sat on the mat"，共 6 个 token：

```
输入:  X_input = [6, 512]    ← 6个token，每个512维
                            ← 第0行: "The" 的嵌入向量
                            ← 第1行: "cat" 的嵌入向量
                            ← ...
输出:  X_enc_out = [6, 512]  ← 序列长度不变
```

每个 Encoder Layer 的输入和输出维度相同，因此可以堆叠 \(N\) 层。

### 2.2 Self-Attention 机制

Self-Attention 让序列中每个位置都能"关注"到所有其他位置。

**Step 1：生成 Q, K, V**

给定输入 \(X \in \mathbb{R}^{T \times d_{model}}\)：

\[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
\]

其中 \(W_Q, W_K \in \mathbb{R}^{d_{model} \times d_k}\)，\(W_V \in \mathbb{R}^{d_{model} \times d_v}\)。

维度变化：

```
X:  [T, d_model] = [6, 512]
Q:  [T, d_k]     = [6, 64]     ← 查询向量
K:  [T, d_k]     = [6, 64]     ← 键向量
V:  [T, d_v]     = [6, 64]     ← 值向量
```

**Step 2：计算注意力分数**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

维度变化：

```
Q @ K^T:     [6, 64] @ [64, 6]  = [6, 6]   ← 6×6 相似度矩阵
                                            ← 第i行第j列 = token i 对 token j 的关注度
÷ sqrt(64):  [6, 6]                        ← 缩放，防止梯度消失
softmax:     [6, 6]                        ← 每行归一化，和为1
× V:         [6, 6] @ [6, 64]   = [6, 64] ← 加权求和结果
```

**直观理解**：

```
                            "The"  "cat"  "sat"  "on"  "the"  "mat"
注意力分数矩阵 (QK^T):        ┌─────────────────────────────┐  ┌─────┐
                    "The"    │ 0.8   0.3   0.2   0.1   0.7  0.2 │  │     │
                    "cat"    │ 0.4   0.9   0.5   0.2   0.3  0.6 │  │ V   │
                    "sat"    │ 0.2   0.6   0.8   0.3   0.2  0.4 │  │ 矩  │
                    "on"     │ 0.1   0.2   0.3   0.9   0.2  0.1 │  │ 阵  │
                    "the"    │ 0.7   0.3   0.2   0.1   0.8  0.3 │  │     │
                    "mat"    │ 0.2   0.5   0.4   0.1   0.3  0.9 │  └─────┘
                             └─────────────────────────────┘
                                    ↓ softmax 每行归一化
                             ┌─────────────────────────────┐
                    "The"    │ 0.25  0.12  0.10  0.08  0.22 0.10│
                    "cat"    │ 0.12  0.28  0.16  0.08  0.10 0.18│
                    "sat"    │ 0.08  0.22  0.28  0.10  0.08 0.14│
                    "on"     │ 0.06  0.08  0.10  0.40  0.06 0.05│
                    "the"    │ 0.22  0.10  0.08  0.06  0.28 0.12│
                    "mat"    │ 0.06  0.16  0.12  0.05  0.10 0.40│
                             └─────────────────────────────┘
                                             ↓ × V
                   输出: [6, 64] ← 每个位置融合了全局信息的表示
```

### 2.3 Multi-Head Attention

将 \(d_{model}\) 均分为 \(h\) 份，每份独立做 Attention，再拼接：

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O
\]

\[
\text{head}_i = \text{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)})
\]

**维度变化**：

```
输入 X: [6, 512]

分头（h=8）:
  head_0: Q_0=[6,64], K_0=[6,64], V_0=[6,64] → Attn → [6,64]
  head_1: Q_1=[6,64], K_1=[6,64], V_1=[6,64] → Attn → [6,64]
  ...
  head_7: Q_7=[6,64], K_7=[6,64], V_7=[6,64] → Attn → [6,64]

拼接:          [6, 64×8] = [6, 512]
× W_O:        [6, 512] @ [512, 512] = [6, 512]  ← 输出，与输入维度相同
```

**不同头学到不同模式**：

```
head_0: "cat" → "sat" (语法依赖，主谓关系)
head_1: "cat" → "mat" (语义关联)
head_2: "The" → "cat" (冠词-名词搭配)
head_3: "on"  ↔ "mat" (介词-名词位置关系)
...
```

### 2.4 位置编码 Positional Encoding

Self-Attention 本身是**置换不变**的——即打乱顺序后注意力分数不变。为了让模型感知位置，需要注入位置信息。

**经典 Sinusoidal 位置编码**：

\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

- \(pos\)：token 在序列中的位置（0-indexed）
- \(i\)：维度索引
- 不同频率的正余弦波，每个位置有唯一的"编码指纹"

**维度变化**：

```
PE: [T, d_model] = [6, 512]    ← 与输入嵌入同维度

最终 Encoder 输入:
  X = token_embedding [6,512] + PE [6,512] → [6,512]
```

**为什么有用**：
- \(PE_{pos}\) 和 \(PE_{pos+k}\) 可以通过线性变换互相表示，让模型能学习到**相对位置**关系
- 可外推到比训练时更长的序列（不需要重新训练）

**可学习位置编码**：直接用 `nn.Embedding(max_len, d_model)` 当作参数训练。

### 2.5 前馈网络 FFN

对每个位置独立应用相同的两层 MLP：

\[
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
\]

**维度变化**：

```
输入:  [6, 512]        ← 每个位置512维
× W_1: [512, 2048]    ← 扩展到4倍
ReLU:  [6, 2048]       ← 非线性激活
× W_2: [2048, 512]    ← 压缩回原始维度
输出:  [6, 512]        ← 维度不变
```

FFN 为每个位置引入**非线性变换**，是 Transformer 中参数量最大的部分。

### 2.6 LayerNorm & 残差连接

每个子层（Self-Attention 或 FFN）后都跟一个残差连接 + LayerNorm：

\[
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
\]

**残差连接**：\(x + \text{Sublayer}(x)\)

- 解决深层网络的梯度消失问题
- 让模型可以堆叠更多层

**LayerNorm**：

\[
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

- 对每个样本、每个 token 独立计算**均值**和**方差**
- 跨特征维度归一化，而非跨 batch

**维度变化**：

```
残差:   [6, 512] + [6, 512]  = [6, 512]    ← 逐元素相加
LayerNorm: [6, 512] → [6, 512]              ← 逐 token 归一化
```

### 2.7 Encoder 完整前向过程

```
输入句子: "The cat sat on the mat"
Token IDs: [1215, 3919, 6153, 1054, 1215, 5270]

                   维度变化                          说明
─────────────────────────────────────────────────────────────────
Token Embedding:   [6, 512]           每个 token 映射为 512 维向量
+ Positional Enc:  [6, 512]           加上位置编码
         ↓
┌─ Encoder Layer 1 ─────────────────────────────────────────┐
│  MultiHead Self-Attention:                                 │
│    Q=X@W_Q, K=X@W_K, V=X@W_V                               │
│    Q=[6,64]×8, K=[6,64]×8, V=[6,64]×8                      │
│    → softmax(QK^T/√64)V → concat → [6,512]                │
│  + 残差 + LayerNorm:    [6,512] → [6,512]                  │
│  FFN: ReLU([6,512]@[512,2048])@[2048,512] → [6,512]       │
│  + 残差 + LayerNorm:    [6,512] → [6,512]                  │
└────────────────────────────────────────────────────────────┘
         ↓  (重复 N=6 层)
┌─ Encoder Layer 6 ─────────────────────────────────────────┐
│  (结构与 Layer 1 相同，但参数不同)                           │
└────────────────────────────────────────────────────────────┘
         ↓
Encoder 输出: X_enc_out = [6, 512]     ← 6个位置，每个512维
                                         ← 供 Decoder 的 Cross-Attention 使用
```

**关键特性**：
- Encoder 可以**并行**处理所有 token（与 RNN 的本质区别）
- 每层输出维度不变，可以任意堆叠层数
- Encoder 可以**一次性看到完整序列**（无 mask）

---

## 三、Decoder 详解

### 3.1 Decoder 的输入与输出

**Decoder 与 Encoder 的结构差异**：

Decoder Layer 包含三个子层：
1. **Masked Self-Attention** — 带因果掩码的自注意力
2. **Cross-Attention** — 查询 Encoder 输出的跨注意力
3. **FFN** — 前馈网络

**训练时的输入**：目标序列（使用 Teacher Forcing）

**推理时的输入**：已生成的部分序列（自回归）

**维度**：

```
Decoder 输入:  [T_dec, d_model]   ← 目标序列嵌入 + 位置编码
Decoder 输出:  [T_dec, d_model]   ← 每步对应一个位置的上下文表示
最终输出:      [T_dec, vocab_size] ← Linear 投影到词表 → softmax
```

### 3.2 Masked Self-Attention & Causal Mask

Decoder 的 Self-Attention 必须保证：**预测第 \(t\) 个位置时，不能看到第 \(t+1\) 及以后的位置**。

**Causal Mask**：

\[
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right) V
\]

\[
M_{ij} = \begin{cases}
0 & i \geq j \quad (\text{允许看当前位置及之前})\\
-\infty & i < j \quad (\text{禁止看未来})
\end{cases}
\]

**数值示例**（序列长度 8）：

```
注意力分数矩阵 QK^T [8,8] (未 mask):
       t0    t1    t2    t3    t4    t5    t6    t7
t0  [ 0.8   0.3   0.2   0.1   0.7   0.2   0.1   0.1]
t1  [ 0.4   0.9   0.5   0.2   0.3   0.6   0.2   0.1]
t2  [ 0.2   0.6   0.8   0.3   0.2   0.4   0.3   0.2]
t3  [ 0.1   0.2   0.3   0.9   0.2   0.1   0.5   0.3]
t4  [ 0.7   0.3   0.2   0.1   0.8   0.3   0.2   0.1]
t5  [ 0.2   0.5   0.4   0.1   0.3   0.9   0.4   0.2]
t6  [ 0.1   0.2   0.3   0.4   0.1   0.3   0.8   0.5]
t7  [ 0.2   0.1   0.2   0.1   0.2   0.1   0.4   0.9]

加 Causal Mask 后 (M_ij = -∞ 当 i < j):
       t0    t1    t2    t3    t4    t5    t6    t7
t0  [ 0.8   -∞    -∞    -∞    -∞    -∞    -∞    -∞]   ← t0 只能看自己
t1  [ 0.4   0.9   -∞    -∞    -∞    -∞    -∞    -∞]   ← t1 只能看 t0,t1
t2  [ 0.2   0.6   0.8   -∞    -∞    -∞    -∞    -∞]
t3  [ 0.1   0.2   0.3   0.9   -∞    -∞    -∞    -∞]
t4  [ 0.7   0.3   0.2   0.1   0.8   -∞    -∞    -∞]
t5  [ 0.2   0.5   0.4   0.1   0.3   0.9   -∞    -∞]
t6  [ 0.1   0.2   0.3   0.4   0.1   0.3   0.8   -∞]
t7  [ 0.2   0.1   0.2   0.1   0.2   0.1   0.4   0.9]  ← t7 能看到全部

softmax 后（-∞ → 0）:
       t0    t1    t2    t3    t4    t5    t6    t7
t0  [ 1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0]
t1  [ 0.38  0.62  0.0   0.0   0.0   0.0   0.0   0.0]
t2  [ 0.18  0.33  0.49  0.0   0.0   0.0   0.0   0.0]
...
```

**关键**：第 \(t\) 行 softmax 后，非零概率只分布在 \([0, t]\) 区间。

### 3.3 Cross-Attention (Encoder-Decoder Attention)

Cross-Attention 是 Decoder 连接 Encoder 的桥梁：

- **Q**：来自 Decoder 的当前隐层（要生成的位置）
- **K, V**：来自 Encoder 的输出（完整输入序列的表示）

\[
\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}
\]

**维度变化**：

```
Decoder 隐层:   X_dec = [8, 512]      ← Decoder Self-Attention 输出
Encoder 输出:   X_enc = [6, 512]      ← Encoder 最终输出

Q = X_dec @ W_Q:   [8, 64] × 8 heads
K = X_enc @ W_K:   [6, 64] × 8 heads
V = X_enc @ W_V:   [6, 64] × 8 heads

QK^T:     [8, 6]   ← Decoder 的 8 个位置 × Encoder 的 6 个位置
                    ← 第i行: "当前生成的token应该关注原文的哪个词？"
softmax:  [8, 6]   ← 每行归一化
× V:      [8, 64]   ← 按权重融合 Encoder 信息
→ concat: [8, 512]
```

**注意**：Cross-Attention **不需要** Causal Mask，因为 Encoder 输出是整个序列，Decoder 可以访问全部。

### 3.4 Decoder 完整前向过程

**训练时**（Teacher Forcing，所有位置并行）：

```
目标序列:           "<sos> 猫 坐 在 垫子 上 <eos>"
Token IDs:         [2, 389, 1205, 87, 4512, 28, 3]   (7个token)

                             维度变化                       说明
────────────────────────────────────────────────────────────────────
Token Embedding + PE:         [7, 512]         目标序列嵌入
         ↓
┌─ Decoder Layer 1 ─────────────────────────────────────────────────┐
│  1. Masked Self-Attention:                                        │
│     Q=K=V=[7,512] → 每头 [7,64] → QK^T=[7,7] → +Mask → softmax   │
│     ×V → concat → [7,512] → 残差+Norm → [7,512]                  │
│                                                                   │
│  2. Cross-Attention:                                              │
│     Q=[7,512] (来自 Decoder), K=V=[6,512] (来自 Encoder)          │
│     QK^T=[7,6] → softmax → ×V → [7,512] → 残差+Norm → [7,512]   │
│                                                                   │
│  3. FFN: [7,512]→[7,2048]→[7,512] → 残差+Norm → [7,512]         │
└───────────────────────────────────────────────────────────────────┘
         ↓ (重复 N=6 层)
         ↓
Linear:  [7, 512] → [7, vocab_size]     ← 投影到词表
Softmax: [7, vocab_size]                 ← 每个位置的概率分布
```

---

## 四、推理过程详解

### 4.1 推理 vs 训练的核心区别

| 方面 | 训练 | 推理 |
|------|------|------|
| **Decoder 输入** | 真实标签（Teacher Forcing） | 模型自身上一步输出 |
| **并行性** | 所有时间步**并行**计算 | 逐个**自回归**生成 |
| **已知信息** | 知道完整的目标序列 | 只知道已经生成的部分 |
| **Encoder** | 每批运行一次 | 每个样本运行一次 |
| **优化技术** | Dropout 等正则化 | KV Cache 加速 |

### 4.2 Encoder 推理

Encoder 在推理时与训练时**完全相同**：

1. 输入完整的源序列
2. 正向传播通过 N 层 Encoder
3. 输出 \(X_{enc\_out} \in \mathbb{R}^{T_{enc} \times d_{model}}\)
4. 输出可**缓存**，供 Decoder 的每一层、每一步重复使用

**数值示例**：

```python
# 推理: 翻译 "The cat sat on the mat" → 中文
# Encoder 只需运行一次

# 输入
src_tokens = tokenize("The cat sat on the mat")  # [6]
src_embed = embedding(src_tokens)                 # [6, 512]

# Encoder 前向 (N=6层)
enc_out = encoder(src_embed)                      # [6, 512]

# 缓存 Encoder 输出，供 Decoder 每一步使用
cache = {"enc_out": enc_out}  # [6, 512] → 不变
```

### 4.3 Decoder 自回归推理

Decoder 推理时逐步生成输出，**每步只生成一个 token**：

```
Step 0:  输入 [<sos>]           → Decoder → 预测 "猫"
Step 1:  输入 [<sos>, 猫]       → Decoder → 预测 "坐"
Step 2:  输入 [<sos>, 猫, 坐]   → Decoder → 预测 "在"
Step 3:  输入 [<sos>, 猫, 坐, 在] → Decoder → 预测 "垫子"
Step 4:  输入 [<sos>, 猫, 坐, 在, 垫子] → Decoder → 预测 "上"
Step 5:  输入 [<sos>, 猫, 坐, 在, 垫子, 上] → Decoder → 预测 "<eos>"
→ 停止
```

**逐维度演示**（以 Step 2 为例）：

```
已生成: [<sos>, 猫, 坐]    (3个token)

1. Token Embedding + PE
   dec_input = embedding([2, 389, 1205]) + PE[:3]   # [3, 512]

2. Masked Self-Attention (只在已生成的 3 个位置之间做):
   Q=K=V = dec_input    # [3, 512]
   QK^T: [3, 3]          # 3×3 注意力矩阵
   + Causal Mask:        # 下三角矩阵
     [ 1.2   -∞    -∞  ]
     [ 0.8   0.6   -∞  ]
     [ 0.5   0.7   0.9 ]
   softmax: [3, 3]
   × V:     [3, 512]

3. Cross-Attention (查询 Encoder 输出):
   Q = MaskedAttn 输出     # [3, 512]
   K = V = enc_out        # [6, 512]  (来自 Encoder)
   QK^T: [3, 6]           # 3个已生成token × 原文6个位置
   softmax: [3, 6]
   × V:     [3, 512]

4. FFN: [3, 512] → [3, 512]

5. Linear + Softmax:
   [3, 512] → [3, vocab_size]
   取最后一步 (第2行) 的概率分布 → 选择 "在"
```

**选择下一个 token 的策略**：

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **Greedy** | 选概率最大的 token | 快速，但可能陷入次优 |
| **Beam Search** | 维护 top-k 条候选序列 | 质量更高，速度稍慢 |
| **Sampling** | 按概率分布随机采样 | 增加多样性 |
| **Top-k / Top-p** | 只在 top-k 或累积概率 p 内采样 | 平衡多样性与质量 |

**Greedy Decoding 示例**：

```
Step 0:  分布: [猫:0.6, 这:0.2, 那:0.1, ...] → 选 "猫" (p=0.6)
Step 1:  分布: [坐:0.5, 在:0.2, 是:0.1, ...] → 选 "坐" (p=0.5)
Step 2:  分布: [在:0.7, 于:0.1, 上:0.1, ...] → 选 "在" (p=0.7)
...
```

**Beam Search (beam=2)**：

```
Step 0:  保留 top-2: ["猫"(0.6), "这"(0.2)]
Step 1:
  从 "猫" 扩展: "猫坐"(0.6×0.5=0.3), "猫在"(0.6×0.2=0.12)
  从 "这" 扩展: "这只"(0.2×0.4=0.08), "这是"(0.2×0.3=0.06)
  保留 top-2: ["猫坐"(0.3), "猫在"(0.12)]
Step 2:
  从 "猫坐" 扩展: "猫坐在"(0.3×0.7=0.21), ...
  从 "猫在" 扩展: "猫在这"(0.12×0.5=0.06), ...
  保留 top-2: ...
...
最终: 选累计概率最高的序列
```

### 4.4 KV Cache 优化

**问题**：自回归推理中，第 \(t\) 步的 Q, K, V 包含了第 \(0\) 到 \(t-1\) 步所有 token 的 K, V。这些 K, V 在前 \(t-1\) 步已经计算过了，每次重新计算浪费巨大。

**KV Cache**：缓存每一步的 K, V，后续步只计算新增 token 的 K, V。

**无 KV Cache**（第 t 步重新计算全部）：

```python
# Step 0: 输入 [<sos>] → 计算 Q0, K0, V0
# Step 1: 输入 [<sos>, 猫] → 重新计算 Q0, K0, V0, Q1, K1, V1 (Q0重复计算!)
# Step 2: 输入 [<sos>, 猫, 坐] → 重新计算 Q0,K0,V0, Q1,K1,V1, Q2,K2,V2
```

计算量随序列长度**平方**增长。

**有 KV Cache**：

```python
# 初始化
K_cache = []   # 累积的 K
V_cache = []   # 累积的 V

# Step 0: 输入 [<sos>]
K_new, V_new = project(embed(<sos>))   # 只算当前位置
K_cache = [K_new];  V_cache = [V_new]  # 缓存
Q = project_q(embed(<sos>))
output = attn(Q, K_cache, V_cache)     # [1, 512]

# Step 1: 输入 [猫]
K_new, V_new = project(embed(猫))       # 只算当前位置
K_cache.append(K_new); V_cache.append(V_new)  # 追加缓存
Q = project_q(embed(猫))
output = attn(Q, K_cache, V_cache)     # [1, 512]

# Step 2: 输入 [坐]
K_new, V_new = project(embed(坐))
K_cache.append(K_new); V_cache.append(V_new)
Q = project_q(embed(坐))
output = attn(Q, K_cache, V_cache)
```

计算量随序列长度**线性**增长。

**维度变化对比**：

```
无 KV Cache:
  Step t: Q = [t+1, 64], K = [t+1, 64], V = [t+1, 64]
          QK^T = [t+1, t+1]    ← 矩阵大小随 t 平方增长
          总计算量 ∝ Σ(t+1)² ≈ O(T³/3)

有 KV Cache:
  Step t: Q = [1, 64]           ← 只算当前位置的 Q
          K_cache = [t+1, 64]   ← 累积的 K
          V_cache = [t+1, 64]   ← 累积的 V
          QK^T = [1, t+1]       ← 矩阵大小只随 t 线性增长
          总计算量 ∝ Σ(t+1) ≈ O(T²/2)
```

**加速效果**（以 \(T=100\) 为例）：

| 方法 | QK^T 总计算量 | 相对比例 |
|------|--------------|---------|
| 无 KV Cache | \(\sum_{t=0}^{99} (t+1)^2 \approx 338,350\) | 100% |
| 有 KV Cache | \(\sum_{t=0}^{99} (t+1) \approx 5,050\) | **1.5%** |

### 4.5 完整推理流程示例

完整翻译推理流程：

```
输入: "The cat sat on the mat"
输出: "猫坐在垫子上"

────────────────────── Step-by-Step ──────────────────────

Encoder (运行1次):
  Input:  "The cat sat on the mat"              → token IDs [6]
  Embed:  [6, 512] + Positional Encoding [6, 512]
  Encoder Layers ×6: [6, 512] → [6, 512]        → enc_out
  缓存: enc_out = [6, 512]

Decoder Step 0:
  Input:       [<sos>]                                    [1 token]
  Embed + PE:  [1, 512]
  Masked SA:   Q=K=V=[1,512] → [1,512]
  Cross Attn:  Q=[1,512], K=V=[6,512] → QK^T=[1,6]       → 关注原文
  FFN:         [1,512]
  Linear:      [1, vocab_size]
  Softmax → 选 "猫"                                      ✓ 第1个词

Decoder Step 1:
  Input:       [<sos>, 猫]                                [2 tokens]
  Embed + PE:  [2, 512]
  Masked SA:   Q=K=V=[2,512] → Mask [2,2] → [2,512]
  Cross Attn:  Q=[2,512], K=V=[6,512] → QK^T=[2,6]
  FFN:         [2,512]
  Linear:      [2, vocab_size]
  取最后一步 → 选 "坐"                                   ✓ 第2个词

Decoder Step 2:
  Input:       [<sos>, 猫, 坐]                            [3 tokens]
  ... 选 "在"                                            ✓ 第3个词

Decoder Step 3:
  ... 选 "垫子"                                          ✓ 第4个词

Decoder Step 4:
  ... 选 "上"                                            ✓ 第5个词

Decoder Step 5:
  ... 选 "<eos>"                                         ✓ 停止

────────────────────── 输出 ──────────────────────
"猫 坐 在 垫子 上"
```

---

## 五、维度汇总与实例

### 5.1 各模块维度速查表

**通用设置**：\(d_{model}=512,\ h=8,\ d_k=d_v=64,\ d_{ff}=2048,\ T_{enc}=10,\ T_{dec}=8\)

| 模块 | 输入维度 | 输出维度 | 参数量（示例） |
|------|---------|---------|--------------|
| **Token Embedding** | [T, vocab_size] → lookup | [T, 512] | vocab×512 |
| **Positional Encoding** | [T] → sin/cos | [T, 512] | 0（固定）或 T×512（可学习） |
| **Encoder Layer** | | | |
| ├ MultiHead Self-Attn | [T, 512] → 8×[T,64] | [T, 512] | 4×512×512=1,048,576 |
| ├ 残差+LayerNorm | [T, 512] | [T, 512] | 2×512=1,024 |
| ├ FFN | [T, 512] | [T, 512] | 512×2048+2048×512=2,097,152 |
| └ 残差+LayerNorm | [T, 512] | [T, 512] | 2×512=1,024 |
| **每层总参数量** | | | ~**3,148,800** |
| **Decoder Layer** | | | |
| ├ Masked Self-Attn | [T, 512] → 8×[T,64] | [T, 512] | 4×512×512=1,048,576 |
| ├ 残差+LayerNorm | [T, 512] | [T, 512] | 1,024 |
| ├ Cross-Attention | Q=[T,512], K/V=[T_enc,512] | [T, 512] | 4×512×512=1,048,576 |
| ├ 残差+LayerNorm | [T, 512] | [T, 512] | 1,024 |
| ├ FFN | [T, 512] | [T, 512] | 2,097,152 |
| └ 残差+LayerNorm | [T, 512] | [T, 512] | 1,024 |
| **每层总参数量** | | | ~**4,197,376** |
| **输出 Linear** | [T, 512] | [T, vocab_size] | 512×vocab_size |

**整体维度流动**：

```
Encoder:
  [T_enc, d_model] = [10, 512]
  ↓  (N 层, 每层维度不变)
  [T_enc, d_model] = [10, 512]  ← 供 Decoder 的 Cross-Attention

Decoder (训练，并行):
  [T_dec, d_model] = [8, 512]
  ↓  (N 层, 每层维度不变)
  [T_dec, d_model] = [8, 512]
  ↓  Linear
  [T_dec, vocab_size] = [8, 30000]

Decoder (推理，自回归第 t 步):
  输入: [1, d_model]           ← 当前 token
  KV Cache: [t, d_model]       ← 累积的 K/V
  输出: [1, d_model]           ← 当前步的上下文
  ↓  Linear → [1, vocab_size] → softmax → 选下一个 token
```

### 5.2 完整数值示例：英译中翻译

**设定**：

- 源语言：英语，目标语言：中文
- 词表大小：30,000
- \(d_{model}=512\)，\(h=8\)，\(N=6\)
- 源句：`"I love transformers"`（3 个 token）
- 目标：`"我爱变压器"`（4 个 token，含 `<sos>` 共 5 个）

**Step 1：Encoder 前向**

```python
# 输入
src_ids = tokenize("I love transformers")        # [3]
src_embed = embedding(src_ids)                    # [3, 512]
src_embed += positional_encoding[:3]              # [3, 512]

# 6层 Encoder
enc_out = src_embed
for layer in encoder_layers:
    # Self-Attention
    attn_out = multihead_self_attn(enc_out)       # [3, 512]
    enc_out = layernorm(enc_out + attn_out)        # [3, 512]
    # FFN
    ffn_out = ffn(enc_out)                         # [3, 512]
    enc_out = layernorm(enc_out + ffn_out)         # [3, 512]

# 最终输出
# enc_out: [3, 512]
# 第0行: "I" 的上下文表示
# 第1行: "love" 的上下文表示
# 第2行: "transformers" 的上下文表示
```

**Step 2：Decoder 训练前向（Teacher Forcing）**

```python
# 目标序列: [<sos>, 我, 爱, 变压器, <eos>] → [5]
tgt_ids = [0, 156, 78, 2341, 1]

tgt_embed = embedding(tgt_ids)                     # [5, 512]
tgt_embed += positional_encoding[:5]               # [5, 512]

# 6层 Decoder
dec_out = tgt_embed
for layer in decoder_layers:
    # 1. Masked Self-Attention
    mask = causal_mask(5)                          # [5, 5], 下三角
    attn_out = masked_multihead_attn(dec_out, mask)# [5, 512]
    dec_out = layernorm(dec_out + attn_out)        # [5, 512]

    # 2. Cross-Attention
    # Q=[5,512], K=V=enc_out=[3,512]
    # QK^T=[5,3] ← 每个目标位置关注源句的3个词
    cross_out = multihead_cross_attn(dec_out, enc_out, enc_out)
    dec_out = layernorm(dec_out + cross_out)       # [5, 512]

    # 3. FFN
    ffn_out = ffn(dec_out)                         # [5, 512]
    dec_out = layernorm(dec_out + ffn_out)         # [5, 512]

# 输出投影
logits = linear(dec_out)                           # [5, 30000]

# 损失计算 (交叉熵)
# 预测词: logits[1] = "我"的概率分布, 标签: "爱"(id=78)
# 预测词: logits[2] = "爱"的概率分布, 标签: "变压器"(id=2341)
# 预测词: logits[3] = "变压器"的概率分布, 标签: "<eos>"(id=1)
```

**Cross-Attention 分数示例**（Decoder Layer 1）：

```
Cross-Attention QK^T [5, 3]:
            "I"    "love"  "transformers"
<sos>       [0.7    0.2     0.1]    ← 起始符主要关注 "I"
我          [0.1    0.8     0.1]    ← "我" 主要关注 "I" 和 "love"
爱          [0.1    0.85    0.05]   ← "爱" 主要关注 "love"
变压器      [0.05   0.15    0.8]    ← "变压器" 关注 "transformers"
<eos>       [0.2    0.3     0.5]    ← 结束符关注整句

→ 每行 softmax 后和为 1
→ 对角线模式：中英文对应关系被正确学到
```

**Step 3：Decoder 推理前向（自回归）**

```python
# Encoder 运行 1 次（同上）
# enc_out = [3, 512] 已缓存

# 自回归生成
generated = [0]  # [<sos>]

for step in range(10):  # 最多生10步
    # 只嵌入最后一个 token
    cur_id = generated[-1]
    cur_embed = embedding(cur_id)                   # [1, 512]
    cur_embed += positional_encoding[step]           # [1, 512]

    # Decoder (使用 KV Cache，只计算最后一步)
    dec_out = cur_embed
    for layer_idx, layer in enumerate(decoder_layers):
        # 1. Masked Self-Attn (用 KV Cache)
        # Q=[1,512], K_cache=[step+1,512], V_cache=[step+1,512]
        attn_out = layer.masked_attn_with_cache(dec_out, k_cache[layer_idx], v_cache[layer_idx])
        dec_out = layernorm(dec_out + attn_out)     # [1, 512]

        # 2. Cross-Attention
        # Q=[1,512], K=V=enc_out=[3,512]
        # QK^T=[1,3] ← 只看当前 token 对原文的关注
        cross_out = layer.cross_attn(dec_out, enc_out, enc_out)
        dec_out = layernorm(dec_out + cross_out)    # [1, 512]

        # 3. FFN
        ffn_out = layer.ffn(dec_out)                # [1, 512]
        dec_out = layernorm(dec_out + ffn_out)      # [1, 512]

    # 预测下一个 token
    logits = linear(dec_out)                        # [1, 30000]
    probs = softmax(logits)                         # [1, 30000]
    next_id = argmax(probs)                         # 取概率最大的

    if next_id == 1:  # <eos>
        break
    generated.append(next_id)

# 输出: [0, 156, 78, 2341, 1]
# 解码: ["<sos>", "我", "爱", "变压器", "<eos>"]
```

---

## 总结

| 主题 | 要点 |
|------|------|
| **Encoder** | 读取完整输入序列 → Self-Attention + FFN → 输出上下文表示 [T_enc, d_model] |
| **Decoder** | Masked Self-Attention → Cross-Attention(查询 Encoder) → FFN → 输出 [T_dec, d_model] |
| **训练** | Teacher Forcing，所有位置并行计算，MSE/交叉熵损失 |
| **推理** | 自回归，逐 token 生成，Encoder 只运行一次，Decoder 逐步运行 |
| **KV Cache** | 缓存历史 K/V，避免重复计算，将推理复杂度从 \(O(T^3)\) 降至 \(O(T^2)\) |
| **输入维度** | 始终为 [序列长度, d_model]，d_model 是统一的隐层维度 |
| **输出维度** | Decoder 最后经过 Linear 投影到 [序列长度, vocab_size]，再 softmax 取概率 |