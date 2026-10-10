"""向量库封装：所有 Chroma 向量库操作（分块/入库/删除/检索）集中在此模块。

使用 LangChain 封装：
- RecursiveCharacterTextSplitter 替代硬切分，优先按段落/句子边界切块
- langchain_chroma.Chroma 作为门面，底层仍是同一个 chroma_db 目录
- 嵌入模型保持 BAAI/bge-small-zh-v1.5，与已有向量数据兼容，无需重建
"""
import os
import threading
import uuid

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import jieba
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

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


# ---------- BM25 关键词索引（混合检索的第二路） ----------
# 内存索引：首次检索时从 ChromaDB 全量重建（懒加载），之后随增删增量维护。
# 与 Chroma 向量库是"一份数据两个视角"：向量按语义找，BM25 按关键词找。
_bm25_lock = threading.Lock()
_bm25_model: BM25Okapi | None = None  # BM25 打分器
_bm25_ids: list[str] = []             # 与语料并行的块 ID（{doc_id}_{idx}）
_bm25_texts: list[str] = []           # 与语料并行的块原文
_bm25_metas: list[dict] = []          # 与语料并行的块 metadata
_RRF_K = 60                           # RRF 融合常数：名次越靠前贡献越大
VECTOR_WEIGHT = 0.6                   # 向量路（语义）权重，工业经验值 0.5~0.7
BM25_WEIGHT = 0.4                     # BM25 路（关键词）权重，两路系数和为 1


def _tokenize(text: str) -> list[str]:
    """jieba 中文分词：BM25 只认"词"，整句需先切成词序列。"""
    return jieba.lcut(text or "")


def _rebuild_bm25() -> None:
    """从 ChromaDB 全量重建 BM25 内存索引（须持有 _bm25_lock）。"""
    global _bm25_model, _bm25_ids, _bm25_texts, _bm25_metas
    found = vector_store.get(include=["documents", "metadatas"])
    _bm25_ids = list(found.get("ids") or [])
    _bm25_texts = list(found.get("documents") or [])
    _bm25_metas = [m or {} for m in (found.get("metadatas") or [])]
    corpus = [_tokenize(t) for t in _bm25_texts]
    _bm25_model = BM25Okapi(corpus) if corpus else None


def _bm25_append(ids: list[str], texts: list[str], metas: list[dict]) -> None:
    """增量加入若干块（须持有 _bm25_lock）。

    索引尚未建立时跳过——之后首次检索会全量重建，数据不丢。
    """
    global _bm25_model
    if _bm25_model is None:
        return
    _bm25_ids.extend(ids)
    _bm25_texts.extend(texts)
    _bm25_metas.extend(metas)
    _bm25_model = BM25Okapi([_tokenize(t) for t in _bm25_texts])


def _bm25_remove(ids: list[str]) -> None:
    """按块 ID 从索引中摘除（须持有 _bm25_lock），与 Chroma 删除同步防脏数据。"""
    global _bm25_model, _bm25_ids, _bm25_texts, _bm25_metas
    if _bm25_model is None:
        return
    gone = set(ids)
    keep = [i for i, cid in enumerate(_bm25_ids) if cid not in gone]
    _bm25_ids = [_bm25_ids[i] for i in keep]
    _bm25_texts = [_bm25_texts[i] for i in keep]
    _bm25_metas = [_bm25_metas[i] for i in keep]
    _bm25_model = BM25Okapi([_tokenize(t) for t in _bm25_texts]) if keep else None


def _bm25_search(
    query: str, k: int, user_id: int, title: str | None = None
) -> list[tuple[str, str]]:
    """BM25 关键词检索，只搜本用户的块，按分数倒序返回 [(块原文, 文档名)]。"""
    hits: list[tuple[str, str]] = []
    with _bm25_lock:
        if _bm25_model is None:
            _rebuild_bm25()  # 懒加载：重启后第一次搜索在这里建索引
        if _bm25_model is not None:
            scores = _bm25_model.get_scores(_tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for i in order:
                if scores[i] <= 0 or len(hits) >= k:
                    break
                meta = _bm25_metas[i] if i < len(_bm25_metas) else {}
                if meta.get("user_id") != user_id:
                    continue  # 别人的块，对不上暗号，跳过
                doc_title = meta.get("title", "未知文档")
                if title and doc_title != title:
                    continue
                hits.append((_bm25_texts[i], doc_title))
    return hits


def chunk_text(content: str) -> list[str]:
    """将文本按边界智能切分为块，content 为空时返回空列表。"""
    if not content:
        return []
    return _splitter.split_text(content)


def add_document_chunks(doc_id: int, title: str, content: str, user_id: int) -> None:
    """将文档内容分块后写入向量库（ID 规则 {doc_id}_{idx} 兼容旧数据）。

    贴标签：每块 metadata 带 user_id，检索时只对上自己的暗号才可见。
    双库同步硬约束：入库后必须同步写入 BM25 关键词索引，防两边数据不一致。
    """
    chunks = chunk_text(content)
    if not chunks:
        return
    ids = [f"{doc_id}_{idx}" for idx in range(len(chunks))]
    metas = [
        {"title": title, "chunk_index": idx, "document_id": doc_id, "user_id": user_id}
        for idx in range(len(chunks))
    ]
    vector_store.add_texts(texts=chunks, metadatas=metas, ids=ids)
    with _bm25_lock:
        _bm25_append(ids, chunks, metas)


def delete_document_chunks(doc_id: int, user_id: int) -> None:
    """从向量库中移除指定文档的全部向量块（删除文档前必须先调用）。

    对暗号：where 两个条件是 AND——文档号和 user_id 都对上才删得到，
    想删别人的文档在这里就查不到任何块。
    全程使用官方公开 API：先按 metadata 查出块 ID，再按 ID 删除。

    双库同步硬约束：删除后必须同步从 BM25 索引摘除，防两边数据不一致。
    """
    found = vector_store.get(
        # Chroma 硬规定：where 放多个条件必须用 $and 包起来（一次只能出现一个操作符），
        # 直接写 {"document_id": x, "user_id": y} 会抛 ValueError 导致删除 500
        where={"$and": [{"document_id": doc_id}, {"user_id": user_id}]}, include=[]
    )
    ids = found.get("ids") or []
    if ids:
        vector_store.delete(ids=ids)
        with _bm25_lock:
            _bm25_remove(ids)


def search_chunks(
    query: str, user_id: int, title: str | None = None, n_results: int | None = None
) -> list[dict]:
    """混合检索：向量（语义）+ BM25（关键词）双路召回，RRF 融合排序。

    强制对暗号：两路召回都只在本用户的块里找，别人的数据不可见。
    返回格式与旧版一致（[{content, title}]），调用方（agent.py）无感知。
    - 向量路：意思相近的块能召回（如"打开文本"→"读取文件"）
    - BM25 路：精确关键词必召回（如专有名词、代码常量、人名编号）
    - RRF 融合：两路都命中的块排最前，任一路失常另一路兜底
    """
    k = n_results or settings.SEARCH_RESULTS
    where_filter: dict = {"user_id": user_id}
    if title:
        where_filter["title"] = title  # Chroma where 多个键自动为 AND

    # 第 1 路：向量语义检索
    vec_docs = vector_store.similarity_search(query, k=k, filter=where_filter)

    # 第 2 路：BM25 关键词检索
    bm_hits = _bm25_search(query, k=k, user_id=user_id, title=title)

    # RRF 加权融合：score = Σ 权重/(RRF_K + 名次)，用 (原文, 文档名) 作为去重键
    # 两路都命中的块拿双份分自动登顶；权重决定"哪路的话更有分量"
    fused: dict[tuple[str, str], float] = {}
    entries: dict[tuple[str, str], dict] = {}
    for rank, doc in enumerate(vec_docs):
        doc_title = doc.metadata.get("title", "未知文档")
        key = (doc.page_content, doc_title)
        fused[key] = fused.get(key, 0.0) + VECTOR_WEIGHT / (_RRF_K + rank + 1)
        entries[key] = {"content": doc.page_content, "title": doc_title}
    for rank, (content, doc_title) in enumerate(bm_hits):
        key = (content, doc_title)
        fused[key] = fused.get(key, 0.0) + BM25_WEIGHT / (_RRF_K + rank + 1)
        entries.setdefault(key, {"content": content, "title": doc_title})

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [entries[key] for key, _ in ranked[:k]]


# ---------- 聊天长期记忆 ----------
# 独立 collection：与文档知识库物理隔离（同目录不同 collection），检索互不干扰
chat_memory_store = Chroma(
    collection_name=settings.CHAT_MEMORY_COLLECTION,
    embedding_function=embedding_func,
    persist_directory=settings.CHROMA_DIR,
)


def add_chat_memory(text: str, user_id: int) -> None:
    """把一条旧问答（一问一答合并的文本）写入长期记忆向量库，贴上 user_id 标签。"""
    if not text:
        return
    chat_memory_store.add_texts(
        texts=[text], metadatas=[{"user_id": user_id}], ids=[uuid.uuid4().hex]
    )


def search_chat_memory(query: str, user_id: int, k: int | None = None) -> list[str]:
    """按语义检索"本用户"最相关的历史问答，按相似度倒序返回前 k 条文本。"""
    docs = chat_memory_store.similarity_search(
        query, k=k or settings.CHAT_MEMORY_RESULTS, filter={"user_id": user_id}
    )
    return [doc.page_content for doc in docs]


def clear_chat_memory(user_id: int) -> None:
    """清空指定用户的聊天长期记忆（使用官方公开 API get + delete）。"""
    found = chat_memory_store.get(where={"user_id": user_id}, include=[])
    ids = found.get("ids") or []
    if ids:
        chat_memory_store.delete(ids=ids)


# ---------- 一次性迁移辅助（老数据没有 user_id 标签，迁移时全量重建用） ----------

def clear_all_document_chunks() -> None:
    """清空文档知识库的全部向量块，并把 BM25 索引重置为待重建状态。"""
    found = vector_store.get(include=[])
    ids = found.get("ids") or []
    if ids:
        vector_store.delete(ids=ids)
    with _bm25_lock:
        global _bm25_model, _bm25_ids, _bm25_texts, _bm25_metas
        _bm25_model = None
        _bm25_ids, _bm25_texts, _bm25_metas = [], [], []


def list_all_chat_memory_texts() -> list[str]:
    """读出全部长期记忆原文（迁移前先备份，防止清空后丢失）。"""
    found = chat_memory_store.get(include=["documents"])
    return [t for t in (found.get("documents") or []) if t]


def clear_all_chat_memory() -> None:
    """清空全部聊天长期记忆（不分用户，仅迁移时使用）。"""
    found = chat_memory_store.get(include=[])
    ids = found.get("ids") or []
    if ids:
        chat_memory_store.delete(ids=ids)
