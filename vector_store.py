"""向量库封装：所有 Chroma 向量库操作（分块/入库/删除/检索）集中在此模块。

使用 LangChain 封装：
- RecursiveCharacterTextSplitter 替代硬切分，优先按段落/句子边界切块
- langchain_chroma.Chroma 作为门面，底层仍是同一个 chroma_db 目录
- 嵌入模型保持 BAAI/bge-small-zh-v1.5，与已有向量数据兼容，无需重建
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# 嵌入模型（与原 SentenceTransformerEmbeddingFunction 同模型同参数，向量空间一致）
embedding_func = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

# 全局唯一的向量库门面实例
vector_store = Chroma(
    collection_name=settings.CHROMA_COLLECTION,
    embedding_function=embedding_func,
    persist_directory=settings.CHROMA_DIR,
)

# 递归文本分割器：优先在 \n\n / \n / 空格 等边界切，避免句子被拦腰截断
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


def chunk_text(content: str) -> list[str]:
    """将文本按边界智能切分为块，content 为空时返回空列表。"""
    if not content:
        return []
    return _splitter.split_text(content)


def add_document_chunks(doc_id: int, title: str, content: str) -> None:
    """将文档内容分块后写入向量库（ID 规则 {doc_id}_{idx} 兼容旧数据）。"""
    chunks = chunk_text(content)
    if not chunks:
        return
    vector_store.add_texts(
        texts=chunks,
        metadatas=[
            {"title": title, "chunk_index": idx, "document_id": doc_id}
            for idx in range(len(chunks))
        ],
        ids=[f"{doc_id}_{idx}" for idx in range(len(chunks))],
    )


def delete_document_chunks(doc_id: int) -> None:
    """从向量库中移除指定文档的全部向量块（删除文档前必须先调用）。

    全程使用官方公开 API：先按 metadata 查出块 ID，再按 ID 删除。
    依赖 API 面：Chroma.get(where=) / Chroma.delete(ids=)，升级 LangChain 时优先验证本函数。
    """
    found = vector_store.get(where={"document_id": doc_id}, include=[])
    ids = found.get("ids") or []
    if ids:
        vector_store.delete(ids=ids)


def search_chunks(query: str, title: str | None = None, n_results: int | None = None) -> list[dict]:
    """按语义检索相关文本块，返回含内容和文档名的字典列表。"""
    where_filter = {"title": title} if title else None
    docs = vector_store.similarity_search(
        query,
        k=n_results or settings.SEARCH_RESULTS,
        filter=where_filter,
    )
    return [
        {"content": doc.page_content, "title": doc.metadata.get("title", "未知文档")}
        for doc in docs
    ]
