# Qwen LoRA 微调示例

基于 **Qwen1.5-1.8B-Chat** 使用 [PEFT](https://github.com/huggingface/peft) 的 LoRA 方法进行参数高效微调（PEFT）的完整示例，包含训练与推理脚本。

## 目录结构

```
Lora/
├── Lora.py                     # 训练脚本（全自动：依赖检查 → 数据准备 → LoRA 微调 → 保存）
├── TestLora.py                 # 推理脚本（加载基础模型 + LoRA 适配器进行生成）
├── model/                      # 基础模型 Qwen1.5-1.8B-Chat（权重需自行下载）
│   └── README.md               # 模型卡说明
└── qwen-lora-finetuned/        # 微调输出目录
    ├── adapter_config.json     # LoRA 适配器配置
    ├── adapter_model.safetensors  # LoRA 适配器权重
    ├── tokenizer*.json         # 分词器
    ├── chat_template.jinja     # 对话模板
    └── checkpoint-6/           # 训练中间检查点（含优化器状态，体积大）
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| [Lora.py](Lora.py) | 训练入口。自动检查并安装依赖，准备 `data.jsonl` 示例数据，使用 8-bit 量化加载 Qwen，在注意力投影层 (`q/k/v/o_proj`) 注入 LoRA 适配器后训练，并保存适配器与分词器。 |
| [TestLora.py](TestLora.py) | 推理验证。加载 `model/` 基础模型并叠加 `qwen-lora-finetuned/` 的 LoRA 适配器，对给定 prompt 生成回复。 |
| `model/` | 基础模型 Qwen1.5-1.8B-Chat，权重文件较大，需从 HuggingFace 下载，**不纳入版本控制**。 |
| `qwen-lora-finetuned/` | 微调结果。适配器权重很小，检查点目录较大。 |

## 环境依赖

```bash
pip install transformers datasets accelerate peft bitsandbytes safetensors torch
```

> 建议 `transformers>=4.37.0`，否则加载 Qwen1.5 可能报 `KeyError: 'qwen2'`。
> `bitsandbytes` 的 8-bit 量化训练通常需要 NVIDIA GPU 环境。

## 使用方法

### 1. 准备基础模型

将 Qwen1.5-1.8B-Chat 下载到 `model/` 目录：

```bash
huggingface-cli download Qwen/Qwen1.5-1.8B-Chat --local-dir model
```

### 2. 训练

```bash
python Lora.py
```

脚本会自动：检查依赖 → 若无 `data.jsonl` 则生成示例数据 → 执行 LoRA 微调 → 将适配器保存到 `qwen-lora-finetuned/`。

如需使用自定义数据，在本目录放置 `data.jsonl`，每行一条 JSON：

```json
{"instruction": "你是谁？", "output": "我是Qwen，一个由阿里云开发的大模型。"}
```

### 3. 推理测试

```bash
python TestLora.py
```

> 注意：[TestLora.py](TestLora.py) 中的 `BASE_DIR` 当前为绝对路径 `e:/AI-Learning/LLM/Lora`，请根据实际路径修改。

## 关键超参数（LoRA）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `r` | 8 | LoRA 秩，越大可训练参数越多 |
| `lora_alpha` | 32 | 缩放系数 |
| `target_modules` | q/k/v/o_proj | 注入适配器的注意力投影层 |
| `lora_dropout` | 0.05 | 降低过拟合 |
| `num_train_epochs` | 3 | 训练轮数 |
| `learning_rate` | 2e-4 | 学习率 |

## 版本控制说明

仓库已配置 [.gitignore](.gitignore)，默认忽略：

- 基础模型权重（`model/*.safetensors` 等大文件）
- 训练检查点 `qwen-lora-finetuned/checkpoint-*/`（含优化器状态）
- 自动生成的 `data.jsonl`
- Python 缓存与虚拟环境

微调后的适配器 `adapter_model.safetensors` 体积很小，默认保留；如不需要上传，可在 `.gitignore` 中取消对应注释。
