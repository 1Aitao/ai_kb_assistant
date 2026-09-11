import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sqlite3
import fitz
from rapidocr_onnxruntime import RapidOCR
import chromadb
from chromadb.utils import embedding_functions

TITLE = "复习题(1)(1)(1).pdf"

print("========== 第一步：OCR 识别 PDF ==========")
ocr = RapidOCR()
doc = fitz.open(os.path.join("uploads", TITLE))
full_text = ""
for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_path = f"_page_{i}.png"
    pix.save(img_path)
    result, _ = ocr(img_path)
    os.remove(img_path)
    if result:
        full_text += "\n".join(line[1] for line in result) + "\n"
    print(f"第 {i+1} 页识别完成，累计文字长度: {len(full_text)}")

full_text = full_text.strip()
print("总文字长度:", len(full_text))
print("前200字:", full_text[:200])

if not full_text:
    print(">>> OCR 也没识别到文字，请确认这个 PDF 里面到底是什么")
    raise SystemExit

print("\n========== 第二步：更新 SQLite ==========")
conn = sqlite3.connect("kb_app.db")
conn.execute("UPDATE documents SET content=? WHERE title=?", (full_text, TITLE))
conn.commit()
row = conn.execute("SELECT id FROM documents WHERE title=?", (TITLE,)).fetchone()
conn.close()
doc_id = row[0]
print(f"SQLite 已更新，document_id={doc_id}")

print("\n========== 第三步：写入向量库 ==========")
func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_or_create_collection(name="documents", embedding_function=func)

try:
    col.delete(where={"title": TITLE})
except Exception:
    pass

chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
for idx, chunk in enumerate(chunks):
    col.add(
        documents=[chunk],
        metadatas=[{"title": TITLE, "chunk_index": idx, "document_id": doc_id}],
        ids=[f"{doc_id}_{idx}"]
    )
print(f"向量库已写入 {len(chunks)} 块，总块数: {col.count()}")