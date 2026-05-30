import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary

# 这是一个简单的前馈神经网络示例，用于二分类（2D点分类）
# 数据：两个高斯分布的点（红/蓝），网络学习把它们分开。

# 1) 生成训练数据
torch.manual_seed(42)
num_samples = 200

# 类别0：中心在 (-1, -1)
class0 = torch.randn(num_samples, 2) * 0.3 + torch.tensor([-1.0, -1.0])
# 类别1：中心在 (1, 1)
class1 = torch.randn(num_samples, 2) * 0.3 + torch.tensor([1.0, 1.0])

# 合并数据与标签
data = torch.cat([class0, class1], dim=0)
labels = torch.cat([torch.zeros(num_samples, dtype=torch.long), torch.ones(num_samples, dtype=torch.long)])

# 打乱顺序
perm = torch.randperm(len(data))
data = data[perm]
labels = labels[perm]

# 2) 定义一个简单的前馈神经网络
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = FeedForwardNet(input_dim=2, hidden_dim=16, output_dim=2)

# 打印模型结构（类似 summary）
print(summary(model, input_size=(1, 2)))

# 3) 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4) 训练
num_epochs = 200
loss_history = []

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_epochs} - loss: {loss.item():.4f}")

# 5) 测试与可视化
with torch.no_grad():
    preds = torch.argmax(model(data), dim=1)
    accuracy = (preds == labels).float().mean().item()
    print(f"训练集准确率: {accuracy * 100:.2f}%")

# 可视化决策边界（可选）
xx, yy = torch.meshgrid(torch.linspace(-2.5, 2.5, 200), torch.linspace(-2.5, 2.5, 200), indexing="xy")
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
with torch.no_grad():
    logits = model(grid)
    probs = torch.softmax(logits, dim=1)[:, 1].reshape(xx.shape)

 
plt.contourf(xx.numpy(), yy.numpy(), probs.numpy(), levels=20, cmap="RdBu", alpha=0.6)
plt.scatter(class0[:, 0].numpy(), class0[:, 1].numpy(), c="blue", edgecolors="k", label="class 0")
plt.scatter(class1[:, 0].numpy(), class1[:, 1].numpy(), c="red", edgecolors="k", label="class 1")
plt.legend()
plt.title("Feedforward NN Decision Boundary")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
