# 本地 RAG 知识库问答

基于 **LangChain + Chroma + Ollama** 的检索增强生成（RAG）示例：将本地 `.txt` 文档向量化存入 Chroma，问答时先检索知识库，命中则结合上下文回答，未命中则回退到大模型自身知识。

## 目录结构

```
RAG/
├── KnowledgeHubBuild.py    # 构建知识库：加载 txt → 切片 → 嵌入 → 存入 Chroma
├── RAGOllama.py            # 交互式问答：检索 Chroma + 调用 Ollama 生成回答
├── X公司.txt               # 知识库源文档
├── 员工手册.txt            # 知识库源文档
└── chroma_db/              # 向量数据库（由脚本自动生成，可重建）
    ├── chroma.sqlite3
    └── <collection-id>/
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| [KnowledgeHubBuild.py](KnowledgeHubBuild.py) | 知识库构建脚本。读取目录下所有 `.txt`，用 `RecursiveCharacterTextSplitter`（chunk 800 / overlap 150）切片，经 `nomic-embed-text` 生成嵌入，持久化到 `chroma_db/`。 |
| [RAGOllama.py](RAGOllama.py) | 交互式问答脚本。对用户提问做相似度检索（top-k=3），分数优于阈值 `0.75` 则把检索内容作为上下文，否则回退通用问答，最终调用本地 Ollama 模型生成回答。 |
| `*.txt` | 知识库源文档，向量化的数据来源。 |
| `chroma_db/` | Chroma 向量库持久化目录，由脚本生成，**不纳入版本控制**（可随时重建）。 |

## 环境依赖

```bash
pip install langchain-text-splitters langchain-ollama langchain-chroma requests
```

并需安装 [Ollama](https://ollama.com/) 且拉取所需模型：

```bash
ollama pull nomic-embed-text     # 嵌入模型
ollama pull qwen3.5:2b           # 生成模型（与 RAGOllama.py 中 MODEL_NAME 对应）
```

> 确保 Ollama 服务已运行在 `http://localhost:11434`。

## 使用方法

### 1. 构建知识库

将待检索的 `.txt` 文档放入本目录，然后运行：

```bash
python KnowledgeHubBuild.py
```

完成后会在 `chroma_db/` 生成向量数据库。

### 2. 启动问答

```bash
python RAGOllama.py
```

在 `你：` 提示符后输入问题，输入 `exit` 或 `quit` 退出。

## 关键参数

| 参数 | 值 | 位置 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | 800 | KnowledgeHubBuild.py | 切片长度 |
| `chunk_overlap` | 150 | KnowledgeHubBuild.py | 切片重叠 |
| 嵌入模型 | `nomic-embed-text` | 两个脚本 | 需与构建/检索保持一致 |
| `MODEL_NAME` | `qwen3.5:2b` | RAGOllama.py | 生成回答的模型 |
| `SIMILARITY_THRESHOLD` | 0.75 | RAGOllama.py | 距离阈值，越小越严格 |
| `k` | 3 | RAGOllama.py | 检索返回的文档数 |

## 版本控制说明

仓库已配置 [.gitignore](.gitignore)，默认忽略：

- 向量数据库 `chroma_db/`（由脚本生成，可重建，体积较大）
- Python 缓存与虚拟环境

知识库源文档 `*.txt` 默认保留上传；若为私有/敏感数据，可在 `.gitignore` 中取消相应注释进行忽略。
