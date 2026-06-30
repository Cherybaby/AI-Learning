import os
import requests
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:2b"

def ask_qwen(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )
    resp.raise_for_status()
    return resp.json()["response"]

SIMILARITY_THRESHOLD = 0.75

while True:
    query = input("你：")
    if query.lower() in ("exit", "quit"):
        break

    results = vectordb.similarity_search_with_score(query, k=3)
    has_knowledge = any(score <= SIMILARITY_THRESHOLD for _, score in results)

    if has_knowledge:
        context = "\n\n".join([doc.page_content for doc, _ in results])

        prompt = (
            "你是一个助手，请**优先使用以下上下文回答问题**。\n"
            "如果上下文不包含答案，你可以使用自己的知识回答。\n\n"
            f"上下文：\n{context}\n\n"
            f"问题：{query}"
        )
        print("\n📚 优先使用知识库")

    else:
        prompt = (
            "你是一个通用助手，请用自己的知识回答以下问题：\n\n"
            f"{query}"
        )
        print("\n🤖 使用大模型知识回答")

    answer = ask_qwen(prompt)
    print("\nQwen：", answer, "\n")