import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# 这是一个最简端到端 Vision-Language（VLA）模型示例。
# 任务：给图像一个简单的“描述”标签（类似图像分类 + 文字输出）。
# 这里我们使用 CIFAR10 数据集的类别名称作为“语言输出”，并用一个 Transformer 解码器将图像特征映射到文本 token 序列。

# -------------- 配置 --------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 5  # 仅输出 5 个 token
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

# -------------- 词表（简单词典 + EOS） --------------
# 这里我们用 CIFAR10 的 10 类标签作为“单词”，并额外加入 <sos>/<eos>。
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 构建词表（word2idx/idx2word）
PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"
vocab = [PAD, SOS, EOS] + CIFAR10_CLASSES
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(vocab)

# -------------- 数据集 --------------
class CIFAR10CaptionDataset(Dataset):
    def __init__(self, train: bool = True):
        from torchvision.datasets import CIFAR10

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.data = CIFAR10(root="./data", train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        # 目标文本序列： <sos> class_name <eos> (固定长度输出)
        tokens = [word2idx[SOS], word2idx[CIFAR10_CLASSES[label]], word2idx[EOS]]
        # pad 到 MAX_SEQ_LEN
        tokens = tokens + [word2idx[PAD]] * (MAX_SEQ_LEN - len(tokens))
        return img, torch.tensor(tokens, dtype=torch.long)

# -------------- 模型 --------------
class VisionLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        # 视觉编码器：用预训练 ResNet18 去掉最后的 fc
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.visual_encoder = resnet

        # 文本 embedding + position embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层：投影到词表
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor):
        # images: (B, 3, H, W)
        # tgt_tokens: (B, T)

        # 视觉特征 (B, C)
        visual_feats = self.visual_encoder(images)  # (B, 512)
        visual_feats = visual_feats.unsqueeze(1)  # (B, 1, 512)

        # 目标 token 嵌入 + 位置嵌入
        token_emb = self.token_embedding(tgt_tokens)  # (B, T, E)
        positions = torch.arange(0, tgt_tokens.size(1), device=tgt_tokens.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)  # (1, T, E)
        tgt = token_emb + pos_emb

        # Transformer 需要 (T, B, E)
        tgt = tgt.permute(1, 0, 2)
        memory = visual_feats.permute(1, 0, 2)  # (1, B, 512)

        # causal mask：防止看到未来 token
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        out = out.permute(1, 0, 2)  # (B, T, E)
        logits = self.output_proj(out)  # (B, T, V)
        return logits

    def greedy_decode(self, images: torch.Tensor, sos_token: int, eos_token: int, max_len: int):
        # 预测：每步用前一步输出生成下一个 token
        B = images.size(0)
        device = images.device
        generated = torch.full((B, max_len), word2idx[PAD], dtype=torch.long, device=device)
        generated[:, 0] = sos_token

        with torch.no_grad():
            for t in range(1, max_len):
                logits = self.forward(images, generated[:, :t])
                next_token = logits.argmax(dim=-1)[:, -1]
                generated[:, t] = next_token

        return generated


# -------------- 训练 --------------
train_ds = CIFAR10CaptionDataset(train=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = VisionLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for images, captions in train_loader:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, captions[:, :-1])  # 预测下一个 token
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - loss: {avg_loss:.4f}")

# -------------- 推理 --------------
model.eval()
with torch.no_grad():
    images, captions = next(iter(train_loader))
    images = images.to(DEVICE)
    generated = model.greedy_decode(images, sos_token=word2idx[SOS], eos_token=word2idx[EOS], max_len=MAX_SEQ_LEN)

    for i in range(5):
        gt_tokens = [idx2word[int(t)] for t in captions[i].tolist() if t != word2idx[PAD]]
        pred_tokens = [idx2word[int(t)] for t in generated[i].tolist() if t != word2idx[PAD]]
        print("GT:", " ".join(gt_tokens))
        print("PRED:", " ".join(pred_tokens))
        print("---")
