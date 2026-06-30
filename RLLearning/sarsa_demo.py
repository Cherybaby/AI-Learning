"""
SARSA 可视化示例（GridWorld）

与 qlearning_demo.py 的主要区别：
- Q-Learning 是 off-policy（目标使用 max_a' Q(s', a')）
- SARSA 是 on-policy（目标使用当前策略实际选到的下一动作 Q(s', a_next)）

这个脚本同样提供：
1) 环境定义
2) epsilon-greedy 训练
3) 贪心回放
4) 训练曲线和策略路径可视化
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GridWorld:
    """简单网格世界环境。"""

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

    def to_pos(self, index: int) -> tuple[int, int]:
        y, x = divmod(index, self.width)
        return x, y


def epsilon_greedy(q_table: np.ndarray, state: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    return int(np.argmax(q_table[state]))


def train_sarsa(
    env: GridWorld,
    episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> tuple[np.ndarray, list[float], list[float]]:
    """训练 SARSA。

    更新公式：
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a_next) - Q(s, a) ]

    其中 a_next 是当前 epsilon-greedy 策略在 s' 上实际选出来的动作。
    """
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    rewards: list[float] = []
    epsilons: list[float] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        epsilons.append(epsilon)

        state = env.reset()
        action = epsilon_greedy(q_table, state, epsilon)
        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, done = env.step(action)

            if done:
                td_target = reward
            else:
                next_action = epsilon_greedy(q_table, next_state, epsilon)
                td_target = reward + gamma * q_table[next_state, next_action]

            q_table[state, action] += alpha * (td_target - q_table[state, action])

            if not done:
                state = next_state
                action = next_action

            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_table, rewards, epsilons


def run_greedy_episode(
    env: GridWorld, q_table: np.ndarray, max_steps: int = 50
) -> tuple[list[tuple[int, int]], float, bool]:
    state = env.reset()
    path = [env.to_pos(state)]
    total_reward = 0.0

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state]))
        state, reward, done = env.step(action)
        path.append(env.to_pos(state))
        total_reward += reward
        if done:
            return path, total_reward, True

    return path, total_reward, False


def render_policy(env: GridWorld, q_table: np.ndarray) -> None:
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    print("\nLearned policy (SARSA):")
    for y in range(env.height):
        row = []
        for x in range(env.width):
            if (x, y) == env.start:
                row.append("S")
            elif (x, y) == env.goal:
                row.append("G")
            elif (x, y) == env.trap:
                row.append("T")
            else:
                s = env._to_index((x, y))
                row.append(arrows[int(np.argmax(q_table[s]))])
        print(" ".join(row))


def _moving_average(values: list[float], window: int = 50) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="valid")


def plot_training_curve(rewards: list[float], epsilons: list[float]) -> None:
    episodes = np.arange(1, len(rewards) + 1)
    ma = _moving_average(rewards, window=50)
    ma_x = np.arange(len(ma)) + (len(rewards) - len(ma) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(episodes, rewards, alpha=0.25, label="Episode reward", color="#1f77b4")
    ax1.plot(ma_x, ma, linewidth=2.0, label="Moving avg (50)", color="#ff7f0e")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("SARSA Training Curve")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, linestyle="--", linewidth=1.2, color="#2ca02c", label="Epsilon")
    ax2.set_ylabel("Epsilon")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()


def plot_policy_and_path(
    env: GridWorld, q_table: np.ndarray, path: list[tuple[int, int]]
) -> None:
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    value_map = np.max(q_table, axis=1).reshape(env.height, env.width)

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
                a = int(np.argmax(q_table[s]))
                ax.text(x, y, arrows[a], ha="center", va="center", color="white", fontsize=11)

    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, color="red", linewidth=2.2, marker="o", markersize=4, label="Greedy path")

    ax.set_title("SARSA Policy and Greedy Path")
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

    env = GridWorld()
    q_table, rewards, epsilons = train_sarsa(env)

    avg_last_100 = float(np.mean(rewards[-100:]))
    print(f"SARSA training finished. Avg reward (last 100 episodes): {avg_last_100:.3f}")

    path, total_reward, done = run_greedy_episode(env, q_table)
    print(f"Greedy run done: {done}, total_reward: {total_reward:.2f}")
    print("Path:", path)

    render_policy(env, q_table)

    plot_training_curve(rewards, epsilons)
    plot_policy_and_path(env, q_table, path)
    plt.show()


if __name__ == "__main__":
    main()
