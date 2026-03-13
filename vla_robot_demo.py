import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 这是一个演示性的“视觉+语言 -> 机器人关节目标”端到端模型。
# 输入：图像 + 文字指令（如“pick bowl”）
# 输出：机器人关节目标角度（比如 6 个电机的目标值）

# ------------- 配置 -------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-3
NUM_JOINTS = 6
MAX_CMD_LEN = 6
EMBED_DIM = 128
HIDDEN_DIM = 256

# ------------- 任务/指令词表 -------------
COMMANDS = [
    "pick bowl",        # 拿碗
    "place bowl",       # 放碗
    "pick cup",         # 拿杯
    "place cup",        # 放杯
    "push button",      # 按按钮
]

# 构造简单的词表（按空格分词）
words = set(w for cmd in COMMANDS for w in cmd.split())
PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"
vocab = [PAD, SOS, EOS] + sorted(words)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(vocab)

# ------------- 伪数据集 -------------
class RobotMotionDataset(Dataset):
    def __init__(self, size: int = 200, image_size=(3, 64, 64)):
        self.size = size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size[1], image_size[2])),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
        ])

        # 为每条命令指定一个固定目标关节角度（演示用）
        self.cmd_to_joint = {
            "pick bowl":  torch.tensor([0.2, -0.5, 0.5, -0.3, 0.1, 0.0]),
            "place bowl": torch.tensor([-0.1, 0.3, -0.4, 0.2, -0.2, 0.1]),
            "pick cup":   torch.tensor([0.3, -0.2, 0.6, -0.1, 0.0, 0.2]),
            "place cup":  torch.tensor([-0.2, 0.4, -0.3, 0.3, -0.1, -0.1]),
            "push button":torch.tensor([0.1, 0.0, 0.2, -0.5, 0.4, -0.2]),
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 模拟一张 RGB 图像（随机噪声）
        img = torch.rand(self.image_size)
        cmd = COMMANDS[idx % len(COMMANDS)]

        # 将命令转换为 token 序列（带 <sos>/<eos> 并 pad）
        tokens = [word2idx[SOS]] + [word2idx[w] for w in cmd.split()] + [word2idx[EOS]]
        while len(tokens) < MAX_CMD_LEN:
            tokens.append(word2idx[PAD])
        tokens = torch.tensor(tokens, dtype=torch.long)

        # 关节目标
        joints = self.cmd_to_joint[cmd]
        return img, tokens, joints


# ------------- 模型 -------------
class VisionLanguageRobot(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_joints: int, max_cmd_len: int):
        super().__init__()
        # 视觉编码器：用小型 ResNet，去掉 fc
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.visual_encoder = resnet

        # 文本编码器（Embedding + GRU）
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx[PAD])
        self.text_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # 融合（视觉 + 语言）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints),
        )

    def forward(self, images: torch.Tensor, cmd_tokens: torch.Tensor):
        # 图像 -> 视觉特征
        vision_feats = self.visual_encoder(images)  # (B, 512)

        # 文本 -> 语言特征
        embedded = self.token_emb(cmd_tokens)  # (B, T, E)
        _, hidden = self.text_encoder(embedded)  # hidden: (1, B, H)
        lang_feats = hidden.squeeze(0)  # (B, H)

        # 融合 -> 关节目标
        fused = torch.cat([vision_feats, lang_feats], dim=-1)
        joints = self.fusion(fused)
        return joints


# ------------- 训练 -------------
dataset = RobotMotionDataset(size=500)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VisionLanguageRobot(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_JOINTS, MAX_CMD_LEN).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for images, cmds, joints in dataloader:
        images = images.to(DEVICE)
        cmds = cmds.to(DEVICE)
        joints = joints.to(DEVICE)

        optimizer.zero_grad()
        pred = model(images, cmds)
        loss = criterion(pred, joints)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch}/{NUM_EPOCHS} - loss: {total_loss / len(dataset):.4f}")


# ------------- 推理 -------------
model.eval()
with torch.no_grad():
    # 用几条示例命令做推理
    test_cmds = ["pick bowl", "place cup", "push button"]
    test_tokens = []
    for cmd in test_cmds:
        tokens = [word2idx[SOS]] + [word2idx[w] for w in cmd.split()] + [word2idx[EOS]]
        while len(tokens) < MAX_CMD_LEN:
            tokens.append(word2idx[PAD])
        test_tokens.append(tokens)

    test_tokens = torch.tensor(test_tokens, dtype=torch.long, device=DEVICE)
    test_imgs = torch.rand((len(test_cmds), 3, 64, 64), device=DEVICE)
    pred_joints = model(test_imgs, test_tokens)

    for cmd, joints in zip(test_cmds, pred_joints):
        print(f"Command: {cmd} -> Predicted joint targets: {joints.tolist()}")
