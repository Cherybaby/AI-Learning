"""
DQN 可视化示例（GridWorld）

特点：
1) 使用神经网络近似 Q(s, a)
2) 经验回放（Replay Buffer）
3) 目标网络（Target Network）稳定训练
4) 与已有示例一致：训练后打印策略并显示图形（不保存图片）
"""

import random
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 自动选择设备：有 GPU 用 GPU，否则 CPU。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    trap: tuple[int, int] = (3, 3)

    def __post_init__(self) -> None:
        self.state = self.start

    @property
    def n_states(self) -> int:
        return self.width * self.height

    @property
    def n_actions(self) -> int:
        # 0: up, 1: right, 2: down, 3: left
        return 4

    def reset(self) -> int:
        self.state = self.start
        return self._to_index(self.state)

    def step(self, action: int) -> tuple[int, float, bool]:
        x, y = self.state

        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            x = min(self.width - 1, x + 1)
        elif action == 2:
            y = min(self.height - 1, y + 1)
        elif action == 3:
            x = max(0, x - 1)

        self.state = (x, y)

        if self.state == self.goal:
            return self._to_index(self.state), 10.0, True
        if self.state == self.trap:
            return self._to_index(self.state), -10.0, True
        return self._to_index(self.state), -0.1, False

    def _to_index(self, pos: tuple[int, int]) -> int:
        x, y = pos
        return y * self.width + x

    def to_pos(self, idx: int) -> tuple[int, int]:
        y, x = divmod(idx, self.width)
        return x, y


class QNetwork(nn.Module):
    """将状态 one-hot 向量映射到每个动作的 Q 值。"""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: deque[tuple[int, int, float, int, bool]] = deque(maxlen=capacity)

    def push(self, transition: tuple[int, int, float, int, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[tuple[int, int, float, int, bool]]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def one_hot_state(state: int, n_states: int) -> np.ndarray:
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


def epsilon_greedy_action(
    q_net: QNetwork,
    state: int,
    n_states: int,
    n_actions: int,
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)

    s = torch.from_numpy(one_hot_state(state, n_states)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = q_net(s)
    return int(torch.argmax(q_values, dim=1).item())


def optimize_dqn(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    n_states: int,
) -> float:
    """从经验回放采样并做一次 DQN 参数更新。"""
    if len(replay) < batch_size:
        return 0.0

    batch = replay.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    s = np.stack([one_hot_state(st, n_states) for st in states]).astype(np.float32)
    ns = np.stack([one_hot_state(st, n_states) for st in next_states]).astype(np.float32)

    s_t = torch.from_numpy(s).to(DEVICE)
    ns_t = torch.from_numpy(ns).to(DEVICE)
    a_t = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    r_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    d_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # 当前 Q(s, a)
    q_pred = q_net(s_t).gather(1, a_t)

    # 目标值：r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        q_next = target_net(ns_t).max(dim=1, keepdim=True)[0]
        q_target = r_t + gamma * q_next * (1.0 - d_t)

    loss = nn.functional.mse_loss(q_pred, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train_dqn(
    env: GridWorld,
    episodes: int = 1200,
    alpha: float = 1e-3,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    target_sync_every: int = 50,
) -> tuple[QNetwork, list[float], list[float], list[float]]:
    q_net = QNetwork(env.n_states, env.n_actions).to(DEVICE)
    target_net = QNetwork(env.n_states, env.n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=alpha)
    replay = ReplayBuffer(capacity=10000)

    rewards: list[float] = []
    epsilons: list[float] = []
    losses: list[float] = []
    epsilon = epsilon_start

    for ep in range(1, episodes + 1):
        epsilons.append(epsilon)
        state = env.reset()
        done = False
        total_reward = 0.0
        ep_losses: list[float] = []

        while not done:
            action = epsilon_greedy_action(
                q_net, state, env.n_states, env.n_actions, epsilon
            )
            next_state, reward, done = env.step(action)
            replay.push((state, action, reward, next_state, done))

            loss = optimize_dqn(
                q_net=q_net,
                target_net=target_net,
                optimizer=optimizer,
                replay=replay,
                batch_size=batch_size,
                gamma=gamma,
                n_states=env.n_states,
            )
            if loss > 0:
                ep_losses.append(loss)

            state = next_state
            total_reward += reward

        if ep % target_sync_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        rewards.append(total_reward)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_net, rewards, epsilons, losses


def run_greedy_episode(
    env: GridWorld, q_net: QNetwork, max_steps: int = 50
) -> tuple[list[tuple[int, int]], float, bool]:
    state = env.reset()
    path = [env.to_pos(state)]
    total_reward = 0.0

    for _ in range(max_steps):
        action = epsilon_greedy_action(
            q_net=q_net,
            state=state,
            n_states=env.n_states,
            n_actions=env.n_actions,
            epsilon=0.0,
        )
        state, reward, done = env.step(action)
        path.append(env.to_pos(state))
        total_reward += reward
        if done:
            return path, total_reward, True

    return path, total_reward, False


def render_policy(env: GridWorld, q_net: QNetwork) -> None:
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    print("\nLearned policy (DQN):")
    for y in range(env.height):
        row = []
        for x in range(env.width):
            pos = (x, y)
            if pos == env.start:
                row.append("S")
            elif pos == env.goal:
                row.append("G")
            elif pos == env.trap:
                row.append("T")
            else:
                s = env._to_index(pos)
                a = epsilon_greedy_action(
                    q_net=q_net,
                    state=s,
                    n_states=env.n_states,
                    n_actions=env.n_actions,
                    epsilon=0.0,
                )
                row.append(arrows[a])
        print(" ".join(row))


def _moving_average(values: list[float], window: int = 50) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="valid")


def plot_training_curves(
    rewards: list[float], epsilons: list[float], losses: list[float]
) -> None:
    episodes = np.arange(1, len(rewards) + 1)
    ma_rewards = _moving_average(rewards, window=50)
    ma_x = np.arange(len(ma_rewards)) + (len(rewards) - len(ma_rewards) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：奖励 + epsilon
    ax1 = axes[0]
    ax1.plot(episodes, rewards, alpha=0.25, label="Episode reward", color="#1f77b4")
    ax1.plot(ma_x, ma_rewards, linewidth=2.0, label="Moving avg (50)", color="#ff7f0e")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("DQN Reward Curve")
    ax1.grid(True, alpha=0.25)

    ax1b = ax1.twinx()
    ax1b.plot(episodes, epsilons, linestyle="--", linewidth=1.2, color="#2ca02c", label="Epsilon")
    ax1b.set_ylabel("Epsilon")

    l1, t1 = ax1.get_legend_handles_labels()
    l2, t2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, t1 + t2, loc="best")

    # 右图：loss
    ax2 = axes[1]
    ax2.plot(episodes, losses, alpha=0.8, color="#d62728", label="Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("DQN Loss Curve")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()


def plot_policy_and_path(env: GridWorld, q_net: QNetwork, path: list[tuple[int, int]]) -> None:
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    # 计算每个格子的 max Q 值用于热力图。
    q_values = []
    for s in range(env.n_states):
        one_hot = torch.from_numpy(one_hot_state(s, env.n_states)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = q_net(one_hot)
        q_values.append(float(torch.max(q).item()))

    value_map = np.array(q_values, dtype=np.float32).reshape(env.height, env.width)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(value_map, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="max Q-value")

    for y in range(env.height):
        for x in range(env.width):
            pos = (x, y)
            if pos == env.start:
                ax.text(x, y, "S", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            elif pos == env.goal:
                ax.text(x, y, "G", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            elif pos == env.trap:
                ax.text(x, y, "T", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            else:
                s = env._to_index(pos)
                a = epsilon_greedy_action(q_net, s, env.n_states, env.n_actions, epsilon=0.0)
                ax.text(x, y, arrows[a], ha="center", va="center", color="white", fontsize=11)

    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, color="red", linewidth=2.2, marker="o", markersize=4, label="Greedy path")

    ax.set_title("DQN Policy and Greedy Path")
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.grid(color="white", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.legend(loc="upper right")

    fig.tight_layout()


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Device: {DEVICE}")

    env = GridWorld()
    q_net, rewards, epsilons, losses = train_dqn(env)

    avg_last_100 = float(np.mean(rewards[-100:]))
    print(f"DQN training finished. Avg reward (last 100 episodes): {avg_last_100:.3f}")

    path, total_reward, done = run_greedy_episode(env, q_net)
    print(f"Greedy run done: {done}, total_reward: {total_reward:.2f}")
    print("Path:", path)

    render_policy(env, q_net)

    plot_training_curves(rewards, epsilons, losses)
    plot_policy_and_path(env, q_net, path)
    plt.show()


if __name__ == "__main__":
    main()
