import math

import torch
import torch.nn as nn


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # embedding特征大小
        self.h = h  # 头的个数
        # 确保d_model可以被h整除
        assert d_model % h == 0, "d_model 不能被 h整除"

        self.d_k = d_model // h  # 每个头特征大小
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # 获取d_k的值。
        d_k = query.shape[-1]
        # Q乘以K的转置，除以根号下d_k。
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 给mask为0的位置填入一个很大的负值，这样在进行softmax，注意力就为0。
            attention_scores.masked_fill_(mask == 0, -1e9)
        # 进行softmax，归一化。得到注意力权重
        # (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # 注意力权重乘以V，得到更新后的embedding。
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # 通过3个全连接层，获取Q、K、V矩阵
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # 对多头进行拆分
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # 计算注意力
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # 多个头合并
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # 乘以输出层
        return self.w_o(x)


# --------------------------------------------------------------------------- #
# Demo：用 Memory Token 场景 + 少量构造数据演示训练与推理过程
#
# 场景类比：q 来自"主序列"（长度 T，比如 T 个待预测的位置/时间步），
#           k/v 来自"memory token"（长度 S，比如 S 个上下文摘要 token，
#           S 通常远小于 T，因为它们是被压缩/浓缩出的少量上下文信息）。
#           这正是 Cross-Attention 的典型用法：q 与 k/v 的序列长度可以不同，
#           只要求最后一维 d_model 相同（多头拆分依赖它）。
#
# 记号:
#   B = batch size（批大小，几组独立样本）
#   T = query 序列长度（主序列，如若干个待推理的位置）
#   S = memory 序列长度（memory token 个数，通常 S << T）
#   D = d_model（特征维度，须能被头数 h 整除）
# --------------------------------------------------------------------------- #

def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_toy_data(B: int, T: int, S: int, D: int, seed: int = 42):
    """构造一组少量的、形状明确的随机数据，模拟"主序列 query + memory token"场景。

    返回:
        q: (B, T, D)  主序列，作为 query（如 T 个待处理的位置/时间步）
        memory: (B, S, D)  memory token，作为 key/value（如 S 个上下文摘要token）
        target: (B, T, D)  一个和 q 形状相同的"伪标签"，仅用于演示训练时的损失计算
    """
    torch.manual_seed(seed)
    q = torch.randn(B, T, D)
    memory = torch.randn(B, S, D)
    # target 仅用于演示 loss 如何计算，不代表真实业务标签；
    # 这里让 target 与 q 存在某种可学习的关联（memory的均值经过线性变换），
    # 使得训练时 loss 确实能随着参数更新而下降，不是纯粹的随机噪声拟合。
    target = q + memory.mean(dim=1, keepdim=True).expand(-1, T, -1) * 0.1
    return q, memory, target


def demo_train_step(block: "MultiHeadAttentionBlock", q, memory, target, lr: float = 1e-2):
    """演示一次完整的训练步骤：前向 -> 计算损失 -> 反向传播 -> 参数更新。

    q/memory 的语义:
        query = q（主序列，形状 (B,T,D)）
        key = value = memory（memory token，形状 (B,S,D)）
    这与 Encoder-Decoder Cross-Attention 的调用方式完全一致：
        block(q, memory, memory, mask=None)
    """
    _print_header("训练阶段（Training）：前向 + 损失 + 反向传播 + 参数更新")

    optimizer = torch.optim.SGD(block.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"输入形状: q(query)={tuple(q.shape)}  memory(key=value)={tuple(memory.shape)}")
    print(f"目标形状: target={tuple(target.shape)}")

    block.train()  # 训练模式：dropout 生效
    losses = []
    n_steps = 5
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()

        # query 来自主序列 q；key/value 来自 memory token（长度 S 与 q 的长度 T 不同）
        output = block(q, memory, memory, mask=None)   # (B, T, D)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  step {step}: output.shape={tuple(output.shape)}  loss={loss.item():.6f}")

    print(f"\n5 步训练后 loss 由 {losses[0]:.6f} 降到 {losses[-1]:.6f}"
          f"（{'下降' if losses[-1] < losses[0] else '未下降，可能需要调整学习率或步数'}，"
          f"说明参数确实在随梯度更新）")
    return losses


def demo_infer_step(block: "MultiHeadAttentionBlock", q, memory):
    """演示一次纯推理：不更新参数，只做前向，并读取真实的注意力权重矩阵。

    关键点：
      - eval() 模式下 dropout 关闭，输出是确定性的（相同输入多次推理结果一致）。
      - attention_scores 的形状是 (B, h, T, S)：每个 query 位置对 S 个 memory token
        的注意力权重，最后一维是 S（memory 长度），不是 T。这是 memory token 场景
        cross-attention 的核心特征——查询序列和被查询序列长度可以不同。
    """
    _print_header("推理阶段（Inference）：纯前向，读取真实注意力权重")

    block.eval()  # 推理模式：关闭 dropout
    with torch.no_grad():
        output = block(q, memory, memory, mask=None)   # (B, T, D)

    attn = block.attention_scores   # (B, h, T, S)，forward 内部保存的真实权重
    B, h, T, S = attn.shape
    print(f"输出形状: {tuple(output.shape)}  （= query 的 B,T 维度 + d_model）")
    print(f"注意力权重形状: {tuple(attn.shape)}  即 (B={B}, h={h}头, T={T}个query位置, S={S}个memory token)")
    print("（最后一维是 S 而不是 T，正是因为 key/value 来自长度为 S 的 memory token）")

    # 验证 eval 模式下确定性：同样输入再跑一次，结果应完全一致
    with torch.no_grad():
        output2 = block(q, memory, memory, mask=None)
    max_diff = (output - output2).abs().max().item()
    print(f"\n确定性核验：两次相同输入的推理输出最大误差 = {max_diff:.2e}（应为0，验证eval模式下无随机性）")

    # 展示第0个batch、第0个头，第一个query位置对所有memory token的注意力权重
    print(f"\n示例：batch=0, head=0, query位置t=0 对 {S} 个 memory token 的注意力权重:")
    print(f"  {attn[0, 0, 0].tolist()}")
    print(f"  权重之和 = {attn[0, 0, 0].sum().item():.6f}（softmax后应≈1.0）")

    return output, attn


if __name__ == "__main__":
    # ---- 构造少量、形状明确的 B/T/S/D 数据 ----
    B, T, S, D, h = 2, 5, 3, 8, 2
    print(f"演示配置: B(批大小)={B}  T(query序列长度)={T}  S(memory token数)={S}  "
          f"D(d_model)={D}  h(头数)={h}  d_k(每头维度)={D // h}")

    q, memory, target = build_toy_data(B, T, S, D)

    block = MultiHeadAttentionBlock(d_model=D, h=h, dropout=0.1)

    # ---- 训练演示：q 作为 query，memory 作为 key/value（cross-attention 用法） ----
    demo_train_step(block, q, memory, target)

    # ---- 推理演示：纯前向，读取真实注意力权重 ----
    demo_infer_step(block, q, memory)