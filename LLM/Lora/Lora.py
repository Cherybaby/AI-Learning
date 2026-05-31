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

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---------- 路径检查 ----------
    if not MODEL_PATH.exists():
        raise RuntimeError(f"❌ 模型路径不存在: {MODEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- Tokenizer ----------
    logger.info("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- 模型 ----------
    logger.info("正在加载模型 (8-bit Quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )

    # ---------- LoRA ----------
    logger.info("配置 LoRA...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    )
    model.print_trainable_parameters()

    # ---------- 数据集 ----------
    logger.info(f"加载数据集: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def preprocess(example):
        prompt = (
            "<|im_start|>system\n你是一个有用的助手。\n<|im_end|>\n"
            f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}"
        )
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0]
        }

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # ---------- 训练参数 ----------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        save_steps=10,
        logging_steps=2,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False
    )

    # ---------- 训练 ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    logger.info("🚀 开始训练...")
    trainer.train()

    # ---------- 保存 ----------
    logger.info("💾 保存模型...")
    model.save_pretrained(OUTPUT_DIR)
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