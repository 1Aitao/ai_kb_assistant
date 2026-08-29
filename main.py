from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import shutil
import os

from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

import models
import schemas
from database import engine, get_db

from fastapi.staticfiles import StaticFiles

import chromadb
from chromadb.utils import embedding_functions

# 加载环境变量（.env 文件）
load_dotenv()

# 读取 API Key 和 Base URL
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 创建大模型客户端
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 初始化 Chroma 向量数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_func
)

# 创建所有数据表
models.Base.metadata.create_all(bind=engine)

# 创建 FastAPI 应用
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 上传文件保存的文件夹
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 查询所有文档
@app.get("/documents", response_model=list[schemas.DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    documents = db.query(models.Document).all()
    return documents

# 创建新文档
@app.post("/documents", response_model=schemas.DocumentResponse)
def create_document(doc: schemas.DocumentCreate, db: Session = Depends(get_db)):
    db_doc = models.Document(title=doc.title, content=doc.content)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

# 查询单个文档
@app.get("/documents/{doc_id}", response_model=schemas.DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return db_doc

# 修改文档
@app.put("/documents/{doc_id}", response_model=schemas.DocumentResponse)
def update_document(doc_id: int, doc: schemas.DocumentCreate, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db_doc.title = doc.title
    db_doc.content = doc.content
    db_doc.updated_at = datetime.now()
    db.commit()
    db.refresh(db_doc)
    return db_doc

# 删除文档
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.delete(db_doc)
    db.commit()
    return {"message": "删除成功"}

# 上传文件
@app.post("/upload")
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    allowed_types = ["text/plain", "text/markdown", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="只支持 .txt .md .pdf 文件")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    content = ""
    if file.content_type == "application/pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            content += page.extract_text() + "\n"
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    db_doc = models.Document(title=file.filename, content=content)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # 把 content 切成每段 500 字
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]

    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"title": file.filename, "chunk_index": idx}],
            ids=[f"{db_doc.id}_{idx}"]
        )
    return {
        "message": "上传成功",
        "filename": file.filename,
        "document_id": db_doc.id
    }

# 聊天接口
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # 1. 检索相关文档
    results = collection.query(
        query_texts=[req.message],
        n_results=3
    )

    # 2. 拼上下文
    context = "\n\n".join(results["documents"][0])

    # 3. 拼 prompt
    prompt = f"请参考以下内容回答问题：\n\n{context}\n\n用户问题：{req.message}"

    # 4. 发给大模型
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个基于知识库回答问题的AI助手。"},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "reply": response.choices[0].message.content
    }
