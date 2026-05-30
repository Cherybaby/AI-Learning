import torch
import torch.nn as nn
import torch.optim as optim

# 简单 LSTM 例子：字符级序列预测（"hello" -> "elloh"）
# 这个例子演示如何用 LSTM 进行序列预测（每一步预测下一个字符）。

# 1) 准备数据
char_set = list("hello")
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

input_str = "hello"
target_str = "elloh"
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

# 输入用独热编码，LSTM 的输入为 (batch, seq, input_size)
input_one_hot = torch.eye(len(char_set))[input_data].unsqueeze(0)  # shape: (1, seq_len, vocab_size)
targets = torch.tensor(target_data).unsqueeze(0)  # shape: (1, seq_len)

# 2) 定义 LSTM 模型
class CharLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden=None):
        # x: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)  # 仍保持 (batch, seq_len, output_size)
        return out, hidden

# 超参数
input_size = len(char_set)
hidden_size = 16
output_size = len(char_set)
num_layers = 1
learning_rate = 0.1
num_epochs = 200

model = CharLSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 3) 训练
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    outputs, _ = model(input_one_hot)
    # CrossEntropyLoss expects (batch * seq_len, classes) 和 (batch * seq_len)
    loss = criterion(outputs.view(-1, output_size), targets.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

# 4) 推理（预测序列）
with torch.no_grad():
    outputs, _ = model(input_one_hot)
    predicted = torch.argmax(outputs, dim=2).squeeze().tolist()
    pred_str = ''.join(idx_to_char[i] for i in predicted)

print("Input:", input_str)
print("Target:", target_str)
print("Predicted:", pred_str)
