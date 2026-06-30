# AI-Learning

一个面向实践的 AI 学习工程，覆盖两条主线：
- DeepLearning：PyTorch 基础与网络结构实验
- LLM：RAG 检索增强问答、LoRA 微调与大模型学习记录

本仓库适合用来做以下事情：
- 快速复现常见深度学习示例
- 体验本地 Ollama + Chroma 的 RAG 流程
- 跑通 Qwen LoRA 微调最小闭环

## 项目导航

- 根目录说明：[README.md](README.md)
- 深度学习模块：[DeepLearning](DeepLearning)
- 大模型模块：[LLM](LLM)

## 目录结构

- [DeepLearning](DeepLearning)
  - [dlLearning.py](DeepLearning/dlLearning.py)：Tensor 基础示例（1D/2D/3D、NumPy 转 Tensor）
  - [RNNDemo.py](DeepLearning/RNNDemo.py)：字符级 RNN 训练示例（hello -> elloh）
  - [lstm_demo.py](DeepLearning/lstm_demo.py)：LSTM 示例
  - [feedforward_demo.py](DeepLearning/feedforward_demo.py)：前馈网络示例
  - [MultiHeadAttentionBlock.py](DeepLearning/MultiHeadAttentionBlock.py)：多头注意力模块实现
  - [vla_demo.py](DeepLearning/vla_demo.py)：视觉语言动作相关实验脚本
  - [vla_robot_demo.py](DeepLearning/vla_robot_demo.py)：视觉+文本到机器人关节目标的端到端演示
  - [Test.ipynb](DeepLearning/Test.ipynb)：实验 Notebook
  - [rnn_model.pth](DeepLearning/rnn_model.pth)：RNN 训练产物

- [LLM](LLM)
  - [LLM_Study.ipynb](LLM/LLM_Study.ipynb)：大模型学习笔记/实验
  - [RAG](LLM/RAG)
    - [KnowledgeHubBuild.py](LLM/RAG/KnowledgeHubBuild.py)：读取 txt 文档、切片、向量化并写入 Chroma
    - [RAGOllama.py](LLM/RAG/RAGOllama.py)：本地检索问答（Ollama 生成 + Chroma 相似度检索）
    - [员工手册.txt](LLM/RAG/员工手册.txt)、[X公司.txt](LLM/RAG/X公司.txt)：示例知识库文档
    - [chroma_db](LLM/RAG/chroma_db)：向量库持久化目录
  - [Lora](LLM/Lora)
    - [Lora.py](LLM/Lora/Lora.py)：Qwen LoRA 微调一体化脚本（依赖检查、样例数据、训练、保存）
    - [TestLora.py](LLM/Lora/TestLora.py)：LoRA 相关测试脚本
    - [model](LLM/Lora/model)：基础模型目录
    - [qwen-lora-finetuned](LLM/Lora/qwen-lora-finetuned)：训练输出（adapter、checkpoint、tokenizer）

## 功能总览

### 1. DeepLearning

适合入门与复习的 PyTorch 练习集合，包括：
- Tensor 基础操作
- RNN/LSTM/前馈网络基础训练
- 注意力机制核心模块实现
- 简化版 VLA（Vision-Language-Action）任务演示

### 2. LLM/RAG

RAG 子模块流程：
1. 通过 [KnowledgeHubBuild.py](LLM/RAG/KnowledgeHubBuild.py) 加载并切分 txt 文档。
2. 使用 Ollama Embedding 模型做向量化并写入 Chroma。
3. 通过 [RAGOllama.py](LLM/RAG/RAGOllama.py) 对用户问题先检索后生成。
4. 当检索分数不满足阈值时，自动回退到通用大模型回答。

### 3. LLM/LoRA

LoRA 子模块提供可直接运行的微调流程：
- 自动检查并安装依赖
- 自动创建示例数据（data.jsonl）
- 基于 Qwen 进行 LoRA 训练
- 保存 adapter 与 tokenizer，便于后续推理复用

## 快速开始

### 1) 环境建议

- Python 3.10+
- 推荐使用虚拟环境
- GPU 可选（LoRA 训练建议有 CUDA 环境）

### 2) 安装常用依赖

在仓库根目录执行：

pip install torch torchvision numpy matplotlib torchinfo
pip install transformers datasets accelerate peft bitsandbytes safetensors
pip install langchain langchain-chroma langchain-ollama chromadb requests

### 3) 运行示例

- DeepLearning 示例：
  - python DeepLearning/dlLearning.py
  - python DeepLearning/RNNDemo.py

- 构建 RAG 知识库：
  - python LLM/RAG/KnowledgeHubBuild.py

- 启动 RAG 对话：
  - python LLM/RAG/RAGOllama.py

- LoRA 微调：
  - python LLM/Lora/Lora.py

## 外部链接

- Ollama: https://ollama.com/
- LangChain: https://python.langchain.com/
- Chroma: https://docs.trychroma.com/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
- PEFT (LoRA): https://huggingface.co/docs/peft/index

## 上传到 GitHub 建议

- 训练产物与向量库通常较大，建议按需管理：
  - [LLM/Lora/qwen-lora-finetuned](LLM/Lora/qwen-lora-finetuned)
  - [LLM/RAG/chroma_db](LLM/RAG/chroma_db)
- 若文件体积较大，建议使用 Git LFS 或在 .gitignore 中忽略训练中间产物。

## 后续可扩展方向

- 增加统一 requirements.txt
- 补充每个子模块的独立 README
- 为 RAG 和 LoRA 增加命令行参数与配置文件
- 添加基础单元测试与 CI
