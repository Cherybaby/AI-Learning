"""
LLM 强化学习（RLHF 思路）最小示例

这个脚本用一个很小的“语言策略模型”模拟大模型在 RLHF 中的训练过程。
核心思想：
1) 有一个初始策略（policy）和冻结参考策略（reference）
2) 模型根据 prompt 生成 answer token
3) 奖励模型给出 reward（这里用规则函数替代）
4) 用 REINFORCE 更新策略，同时加入 KL 惩罚约束不偏离参考模型

说明：
- 这是教学示例，不是工业级 RLHF 框架。
- 重点是展示“奖励优化 + KL 约束”的训练逻辑。
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PromptItem:
    """一条训练样本：prompt 及其期望答案。"""

    prompt: str
    correct_answer: str


class TinyPolicyLM(nn.Module):
    """极简策略模型：prompt_id -> hidden -> answer token logits。

    这里用 Embedding + Linear 模拟语言模型的“条件分布输出”：
    - 输入：一个 prompt 的离散 id
    - 输出：候选答案词表上的 logits

    在真实 LLM 中，这一步通常由 Transformer 完成，并在更大词表上输出。
    """

    def __init__(self, n_prompts: int, n_answers: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_prompts, hidden_size)
        self.head = nn.Linear(hidden_size, n_answers)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        # prompt_ids: [batch]
        h = self.embed(prompt_ids)
        # h: [batch, hidden_size]
        logits = self.head(h)
        # logits: [batch, n_answers]
        return logits


def build_dataset() -> list[PromptItem]:
    """构造一个很小的“指令-答案”集合，用于演示 RLHF 的训练闭环。"""
    return [
        PromptItem(prompt="2+2=?", correct_answer="4"),
        PromptItem(prompt="3+5=?", correct_answer="8"),
        PromptItem(prompt="中国首都是?", correct_answer="北京"),
        PromptItem(prompt="法国首都是?", correct_answer="巴黎"),
        PromptItem(prompt="天空通常是什么颜色?", correct_answer="蓝色"),
    ]


def reward_model(answer: str, correct_answer: str) -> float:
    """规则型奖励函数（替代真实 reward model）。

    真实 RLHF 通常是单独训练的 Reward Model（RM），
    会输入 prompt+answer 并输出标量偏好分数。
    这里为了教学可控，直接用规则打分。
    """
    if answer == correct_answer:
        return 1.0
    # 轻微语义相近奖励（演示用）
    if answer in ["青色", "浅蓝", "蓝"] and correct_answer == "蓝色":
        return 0.4
    return -0.2


def moving_average(values: list[float], window: int = 30) -> np.ndarray:
    """滑动平均，平滑波动，便于观察训练趋势。"""
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="valid")


def evaluate_accuracy(
    model: TinyPolicyLM,
    prompt_to_id: dict[str, int],
    id_to_answer: dict[int, str],
    dataset: list[PromptItem],
) -> float:
    """用贪心解码评估当前策略在数据集上的准确率。"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in dataset:
            # 单样本推理：prompt -> logits -> argmax(answer_id)
            p = torch.tensor([prompt_to_id[item.prompt]], dtype=torch.long, device=DEVICE)
            logits = model(p)
            pred_id = int(torch.argmax(logits, dim=-1).item())
            pred_answer = id_to_answer[pred_id]
            if pred_answer == item.correct_answer:
                correct += 1
    return correct / len(dataset)


def train_rlhf_toy(
    steps: int = 1200,
    lr: float = 3e-2,
    kl_coef: float = 0.08,
    seed: int = 42,
) -> None:
    """训练主循环：REINFORCE + KL(reference) + baseline。

    目标函数（最大化）近似为：
      E[ advantage * log pi(a|s) - kl_coef * KL(pi || pi_ref) ]

    代码中通过最小化负号后的 loss 实现。
    """

    # 固定随机种子，便于复现实验曲线。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = build_dataset()

    prompts = [x.prompt for x in dataset]
    # 加入一些“干扰答案”，让策略确实需要学习筛选。
    answers = sorted(list({x.correct_answer for x in dataset} | {"4", "8", "上海", "伦敦", "蓝色", "青色"}))

    prompt_to_id = {p: i for i, p in enumerate(prompts)}
    answer_to_id = {a: i for i, a in enumerate(answers)}
    id_to_answer = {i: a for a, i in answer_to_id.items()}

    policy = TinyPolicyLM(n_prompts=len(prompts), n_answers=len(answers)).to(DEVICE)

    # 参考策略：冻结不训练，用于提供 KL 约束锚点。
    # 作用：防止策略在奖励驱动下发散或发生“语言崩塌”。
    ref_policy = TinyPolicyLM(n_prompts=len(prompts), n_answers=len(answers)).to(DEVICE)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_hist: list[float] = []
    acc_hist: list[float] = []
    loss_hist: list[float] = []

    # 运行平均作为 baseline，降低 REINFORCE 方差。
    baseline = 0.0

    for step in range(1, steps + 1):
        policy.train()

        # 1) 随机采样一条 prompt
        item = random.choice(dataset)
        prompt_id = torch.tensor([prompt_to_id[item.prompt]], dtype=torch.long, device=DEVICE)

        # 2) 当前策略分布
        logits = policy(prompt_id)
        # 使用分类分布从答案词表采样，模拟“生成一个 token”。
        dist = torch.distributions.Categorical(logits=logits)

        # 3) 采样 answer token（模拟生成）
        sampled_answer_id = dist.sample()  # shape: [1]
        # log pi(a|s)，用于策略梯度项。
        logp = dist.log_prob(sampled_answer_id)  # shape: [1]

        sampled_answer = id_to_answer[int(sampled_answer_id.item())]
        reward = reward_model(sampled_answer, item.correct_answer)

        # 4) KL 惩罚：约束策略不要偏离 reference 太快
        with torch.no_grad():
            ref_logits = ref_policy(prompt_id)
            ref_probs = torch.softmax(ref_logits, dim=-1)

        policy_log_probs = torch.log_softmax(logits, dim=-1)
        policy_probs = torch.softmax(logits, dim=-1)
        # KL(pi || pi_ref) = sum pi * (log pi - log pi_ref)
        # 这里是按当前单样本 prompt 计算的一维 KL。
        kl = torch.sum(policy_probs * (policy_log_probs - torch.log(ref_probs + 1e-8)), dim=-1)

        # 5) REINFORCE + baseline + KL
        # baseline 采用指数滑动平均，降低高方差采样带来的训练抖动。
        baseline = 0.95 * baseline + 0.05 * reward
        advantage = reward - baseline

        # maximize advantage * logp - kl_coef * KL
        # => minimize negative objective
        loss = -(advantage * logp) + kl_coef * kl
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_hist.append(reward)
        loss_hist.append(float(loss.item()))

        # 准确率不必每步算，定期评估即可（降低开销）。
        if step % 20 == 0:
            acc = evaluate_accuracy(policy, prompt_to_id, id_to_answer, dataset)
            acc_hist.append(acc)
        else:
            acc_hist.append(acc_hist[-1] if acc_hist else 0.0)

        if step % 200 == 0:
            print(f"Step {step:4d} | reward(avg100)={np.mean(reward_hist[-100:]):.3f} | acc={acc_hist[-1]:.3f}")

    final_acc = evaluate_accuracy(policy, prompt_to_id, id_to_answer, dataset)
    print(f"Final accuracy: {final_acc:.3f}")

    # 打印最终预测
    print("\nFinal predictions:")
    policy.eval()
    with torch.no_grad():
        for item in dataset:
            p = torch.tensor([prompt_to_id[item.prompt]], dtype=torch.long, device=DEVICE)
            logits = policy(p)
            pred_id = int(torch.argmax(logits, dim=-1).item())
            pred = id_to_answer[pred_id]
            print(f"Prompt: {item.prompt:<12s} | Pred: {pred:<4s} | GT: {item.correct_answer}")

    # 可视化（仅显示，不保存文件）
    x = np.arange(1, len(reward_hist) + 1)
    ma_reward = moving_average(reward_hist, window=50)
    ma_x = np.arange(len(ma_reward)) + (len(reward_hist) - len(ma_reward) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1 = axes[0]
    ax1.plot(x, reward_hist, alpha=0.25, label="Reward per step", color="#1f77b4")
    ax1.plot(ma_x, ma_reward, linewidth=2, label="Moving avg (50)", color="#ff7f0e")
    ax1.set_title("RLHF Toy Reward Curve")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2 = axes[1]
    # accuracy 看策略质量，loss 看优化行为（两者一起观察更完整）。
    ax2.plot(x, acc_hist, color="#2ca02c", label="Greedy accuracy")
    ax2.plot(x, loss_hist, color="#d62728", alpha=0.5, label="Loss")
    ax2.set_title("Accuracy / Loss")
    ax2.set_xlabel("Training step")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    plt.show()


def main() -> None:
    """入口函数。"""
    print(f"Device: {DEVICE}")
    train_rlhf_toy()


if __name__ == "__main__":
    main()
