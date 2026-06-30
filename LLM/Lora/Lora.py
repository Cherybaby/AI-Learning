import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import subprocess
import json
import torch
from pathlib import Path

# ==================================================
# ✅ 全局配置（必须放在最外面）
# ==================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "qwen-lora-finetuned"

# ==================================================
# 第一步：环境准备与依赖检查
# ==================================================
def check_and_install_dependencies():
    required_packages = [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors"
    ]
    print("🔍 正在检查依赖库...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} 已安装")
        except ImportError:
            print(f"  ❌ {package} 未安装，正在安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ==================================================
# 第二步：准备数据文件
# ==================================================
def create_sample_data():
    data_file = Path("./data.jsonl")
    if data_file.exists():
        print("✅ 检测到 data.jsonl 文件，直接使用。")
        return str(data_file)

    print("⚠️ 未检测到 data.jsonl，正在创建示例数据...")
    sample_data = [
        {"instruction": "你是谁？", "output": "我是Qwen，一个由阿里云开发的大模型。"},
        {"instruction": "介绍一下深度学习", "output": "深度学习是机器学习的一个分支。"},
        {"instruction": "1+1等于多少？", "output": "1+1等于2。"},
        {"instruction": "什么是RAG？", "output": "RAG 是检索增强生成。"}
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 示例数据已创建: {data_file}")
    return str(data_file)

# ==================================================
# 第三步：执行微调
# ==================================================
def start_finetuning(data_path):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
        Trainer
    )
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import logging

    # 配置全局日志级别为 INFO，便于观察训练流程。
    logging.basicConfig(level=logging.INFO)
    # 获取当前模块 logger，用于统一输出日志信息。
    logger = logging.getLogger(__name__)

    # ---------- 路径检查 ----------
    if not MODEL_PATH.exists():
        raise RuntimeError(f"❌ 模型路径不存在: {MODEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- Tokenizer ----------
    logger.info("正在加载 Tokenizer...")
    # trust_remote_code=True 允许加载模型仓库中的自定义代码。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # 许多 CausalLM 默认无 pad_token，训练时通常回退到 eos_token。
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- 模型 ----------
    logger.info("正在加载模型 (8-bit Quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        # 8bit 量化显著降低显存占用，适合消费级显卡训练。
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        # 自动把模型层分配到可用设备（如单卡/多卡）。
        device_map="auto",
        # 半精度权重可进一步减少显存并提升吞吐。
        dtype=torch.float16,
        trust_remote_code=True
    )

    # ---------- LoRA ----------
    logger.info("配置 LoRA...")
    # 为 k-bit 训练做预处理（如冻结参数、启用梯度检查点相关兼容设置）。
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            # r 越大可训练参数越多，表达能力更强但开销更大。
            r=8,
            # LoRA 缩放系数，影响增量权重的更新幅度。
            lora_alpha=32,
            # 指定在注意力投影层上注入 LoRA 适配器。
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            # 训练时对 LoRA 分支使用 dropout，降低过拟合风险。
            lora_dropout=0.05,
            # bias 不参与 LoRA 训练，减少额外参数。
            bias="none",
            # 指明任务类型为自回归语言建模。
            task_type="CAUSAL_LM"
        )
    )
    # 打印可训练参数占比，确认是否仅训练 LoRA 层。
    model.print_trainable_parameters()

    # ---------- 数据集 ----------
    logger.info(f"加载数据集: {data_path}")
    # 从 json/jsonl 文件构建训练集 split。
    dataset = load_dataset("json", data_files=data_path, split="train")

    def preprocess(example):
        # 组装 Qwen 对话模板：system + user + assistant。
        prompt = (
            "<|im_start|>system\n你是一个有用的助手。\n<|im_end|>\n"
            f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}"
        )
        tokenized = tokenizer(
            prompt,
            # 过长样本截断到 max_length 以内。
            truncation=True,
            max_length=512,
            # 定长 padding，方便组成批次。
            padding="max_length",
            # 返回 PyTorch Tensor 以供 Trainer 直接使用。
            return_tensors="pt"
        )
        return {
            # 训练输入 token ids。
            "input_ids": tokenized["input_ids"][0],
            # attention mask 标注有效 token 与 padding。
            "attention_mask": tokenized["attention_mask"][0],
            # CausalLM 训练通常 labels 与 input_ids 对齐。
            "labels": tokenized["input_ids"][0]
        }

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # ---------- 训练参数 ----------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        # 梯度累积 4 步后再更新一次，等效放大批大小。
        gradient_accumulation_steps=4,
        # bitsandbytes 的 8bit AdamW 优化器。
        optim="paged_adamw_8bit",
        # 每 10 步保存一次检查点。
        save_steps=10,
        logging_steps=2,
        learning_rate=2e-4,
        # 启用 FP16 混合精度训练。
        fp16=True,
        # 最多保留 2 个检查点，避免磁盘占满。
        save_total_limit=2,
        # 关闭 wandb/tensorboard 等外部上报。
        report_to="none",
        # 不删除未被模型 forward 显式使用的列。
        remove_unused_columns=False
    )

    # ---------- 训练 ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        # 传入已经 token 化后的训练集。
        train_dataset=tokenized_dataset
    )

    logger.info("🚀 开始训练...")
    trainer.train()

    # ---------- 保存 ----------
    logger.info("💾 保存模型...")
    # 保存 LoRA 适配器权重与相关配置。
    model.save_pretrained(OUTPUT_DIR)
    # 同步保存分词器，推理时保持词表与特殊 token 一致。
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"✅ 训练完成！模型保存在: {OUTPUT_DIR}")

# ==================================================
# 主程序入口
# ==================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Qwen 模型微调脚本 (全自动版)")
    print("=" * 50)

    check_and_install_dependencies()
    data_file = create_sample_data()
    start_finetuning(data_file)

    print("\n脚本执行结束。")