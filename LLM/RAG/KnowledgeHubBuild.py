import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 当前 py 文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# 自定义加载器
def load_text_files(directory):
    texts = []
    metadatas = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            path = os.path.join(directory, file)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                metadatas.append({"source": path})
    return texts, metadatas

# 2. 知识库路径
KB_DIR = BASE_DIR

if not os.path.exists(KB_DIR):
    raise FileNotFoundError(f"知识库路径 {KB_DIR} 不存在")

# 3. 加载 txt
texts, metadatas = load_text_files(KB_DIR)

if not texts:
    raise ValueError("未找到任何 .txt 文件")

# 4. 切片
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = []
metas = []
for text, meta in zip(texts, metadatas):
    splits = splitter.split_text(text)
    chunks.extend(splits)
    metas.extend([meta] * len(splits))

# 5. 嵌入
embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 6. 存储（✅ 使用绝对路径）
vectordb = Chroma.from_texts(
    texts=chunks,
    metadatas=metas,
    embedding=embedding,
    persist_directory=CHROMA_DIR
)

print("✅ 知识库构建完成！")

