from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_DIR = "e:/AI-Learning/LLM/Lora"
MODEL_PATH = f"{BASE_DIR}/model"
LORA_PATH = f"{BASE_DIR}/qwen-lora-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

prompt = "什么是RAG？"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))