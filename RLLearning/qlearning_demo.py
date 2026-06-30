"""
Q-Learning 可视化示例（GridWorld）

这个文件演示了一个完整的强化学习最小闭环：
1) 定义环境（状态、动作、奖励、终止条件）
2) 用 epsilon-greedy 进行探索/利用平衡
3) 使用 Q-Learning 迭代更新 Q 表
4) 训练后做贪心策略回放
5) 保存训练曲线与策略路径图
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GridWorld:
    """简单网格世界环境。

    网格坐标约定：
    - 左上角是 (0, 0)
    - x 向右增大，y 向下增大

    终止条件：
    - 到达 goal: 奖励 +10，回合结束
    - 踩到 trap: 奖励 -10，回合结束
    - 其它格子: 每步 -0.1（鼓励更短路径）
    """

    width: int = 5
    height: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    trap: tuple[int, int] = (3, 3)

    def __post_init__(self) -> None:
        # dataclass 初始化字段后，把当前状态重置到起点。
        self.state = self.start

    @property
    def n_states(self) -> int:
        # 状态数 = 网格总格子数。
        return self.width * self.height

    @property
    def n_actions(self) -> int:
        # 动作编码：0 上，1 右，2 下，3 左
        return 4

    def reset(self) -> int:
        """重置环境并返回起点的状态索引。"""
        self.state = self.start
        return self._to_index(self.state)

    def step(self, action: int) -> tuple[int, float, bool]:
        """执行一步动作，返回 (next_state, reward, done)。

        参数:
        - action: 离散动作 id（0~3）

        返回:
        - next_state: 下一状态索引
        - reward: 本步即时奖励
        - done: 是否终止回合
        """
        x, y = self.state

        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # right
            x = min(self.width - 1, x + 1)
        elif action == 2:  # down
            y = min(self.height - 1, y + 1)
        elif action == 3:  # left
            x = max(0, x - 1)

        self.state = (x, y)

        if self.state == self.goal:
            return self._to_index(self.state), 10.0, True
        if self.state == self.trap:
            return self._to_index(self.state), -10.0, True

        # 非终点/陷阱：给予小负奖励，迫使智能体更快到终点。
        return self._to_index(self.state), -0.1, False

    def _to_index(self, pos: tuple[int, int]) -> int:
        """把二维坐标 (x, y) 映射到一维状态索引。"""
        x, y = pos
        return y * self.width + x

    def to_pos(self, index: int) -> tuple[int, int]:
        """把一维状态索引映射回二维坐标 (x, y)。"""
        y, x = divmod(index, self.width)
        return x, y


def epsilon_greedy(q_table: np.ndarray, state: int, epsilon: float) -> int:
    """epsilon-greedy 选动作。

    - 以 epsilon 概率随机探索
    - 以 1-epsilon 概率选择当前 Q 值最大的动作（利用）
    """
    if random.random() < epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    return int(np.argmax(q_table[state]))


def train_q_learning(
    env: GridWorld,
    episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> tuple[np.ndarray, list[float], list[float]]:
    """训练 Q-Learning。

    Q-Learning 更新公式：
    Q(s, a) <- Q(s, a) + alpha * ( td_target - Q(s, a) )
    其中
    td_target = r + gamma * max_a' Q(s', a')    (非终止)
    td_target = r                                 (终止)

    返回:
    - q_table: 训练后的 Q 表
    - rewards: 每回合累计奖励
    - epsilons: 每回合的 epsilon（用于可视化探索衰减）
    """
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    rewards: list[float] = []
    epsilons: list[float] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        epsilons.append(epsilon)
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done = env.step(action)

            # 目标值 = 即时奖励 + 折扣后的下一状态最优价值（终止时后项置 0）。
            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q * (0.0 if done else 1.0)

            # 沿 TD 误差方向更新当前状态-动作的 Q 值。
            q_table[state, action] += alpha * (td_target - q_table[state, action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_table, rewards, epsilons


def run_greedy_episode(env: GridWorld, q_table: np.ndarray, max_steps: int = 50) -> tuple[list[tuple[int, int]], float, bool]:
    """使用训练好的 Q 表进行一次纯贪心回放。

    用途：直观看训练后策略是否能稳定到达目标。
    """
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
    """在终端打印策略箭头图（文本版）。"""
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    print("\nLearned policy:")
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
    """计算滑动平均，用于平滑训练曲线。"""
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.array(values, dtype=np.float64), kernel, mode="valid")


def plot_training_curve(rewards: list[float], epsilons: list[float]) -> None:
    """绘制训练过程图。

    左轴：每回合奖励 + 奖励移动平均
    右轴：epsilon 衰减
    """
    episodes = np.arange(1, len(rewards) + 1)
    ma = _moving_average(rewards, window=50)
    ma_x = np.arange(len(ma)) + (len(rewards) - len(ma) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(episodes, rewards, alpha=0.25, label="Episode reward", color="#1f77b4")
    ax1.plot(ma_x, ma, linewidth=2.0, label="Moving avg (50)", color="#ff7f0e")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Q-Learning Training Curve")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, linestyle="--", linewidth=1.2, color="#2ca02c", label="Epsilon")
    ax2.set_ylabel("Epsilon")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()


def plot_policy_and_path(env: GridWorld, q_table: np.ndarray, path: list[tuple[int, int]]) -> None:
    """绘制策略热力图 + 贪心路径图。

    底图是每个状态的 max Q 值；
    叠加箭头表示该状态下的贪心动作；
    红线显示一次贪心回放实际走过的路径。
    """
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    value_map = np.max(q_table, axis=1).reshape(env.height, env.width)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(value_map, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="max Q-value")

    # 在每个非终止格子叠加贪心动作箭头，起点/终点/陷阱显示标记字符。
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
    # 红线表示回放轨迹，便于验证策略是否绕开陷阱并到达终点。
    ax.plot(px, py, color="red", linewidth=2.2, marker="o", markersize=4, label="Greedy path")

    ax.set_title("Learned Policy and Greedy Path")
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.grid(color="white", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.legend(loc="upper right")

    fig.tight_layout()


def main() -> None:
    """主流程：训练 -> 评估 -> 终端展示 -> 图形显示。"""
    # 固定随机种子，保证每次运行结果可复现。
    random.seed(42)
    np.random.seed(42)

    env = GridWorld()
    q_table, rewards, epsilons = train_q_learning(env)

    # 用最后 100 回合平均奖励衡量收敛质量。
    avg_last_100 = float(np.mean(rewards[-100:]))
    print(f"Training finished. Avg reward (last 100 episodes): {avg_last_100:.3f}")

    path, total_reward, done = run_greedy_episode(env, q_table)
    print(f"Greedy run done: {done}, total_reward: {total_reward:.2f}")
    print("Path:", path)

    # 先给出终端版本策略，方便无图形环境快速查看。
    render_policy(env, q_table)

    # 仅显示图形，不保存到文件。
    plot_training_curve(rewards, epsilons)
    plot_policy_and_path(env, q_table, path)
    plt.show()


if __name__ == "__main__":
    main()
