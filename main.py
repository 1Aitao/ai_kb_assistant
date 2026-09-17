"""路由层：HTTP 接口定义。业务逻辑下沉至 agent/vector_store/extractors 各层。"""
import os
import shutil
import uuid
import hashlib
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import models
import schemas
from agent import run_chat
from config import settings
from database import engine, get_db
from extractors import EXTRACTORS, warm_up_ocr
from vector_store import add_document_chunks, chunk_text, delete_document_chunks

models.Base.metadata.create_all(bind=engine)

# 服务启动时预热 OCR 引擎，避免重启后第一次上传扫描件额外慢约 50%
warm_up_ocr()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = settings.UPLOAD_DIR


@app.get("/")
def home():
    return RedirectResponse(url="/static/index.html")


# ---------- 文档 CRUD ----------

@app.get("/documents", response_model=list[schemas.DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    documents = db.query(models.Document).all()
    return documents


@app.post("/documents", response_model=schemas.DocumentResponse)
def create_document(doc: schemas.DocumentCreate, db: Session = Depends(get_db)):
    db_doc = models.Document(title=doc.title, content=doc.content)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


@app.get("/documents/{doc_id}", response_model=schemas.DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return db_doc


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

    # 同步向量库：先清除旧块，再按新内容重建，避免检索到过期内容
    delete_document_chunks(db_doc.id)
    add_document_chunks(db_doc.id, db_doc.title, db_doc.content)
    return db_doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 同步向量库：先清向量块，再删数据库记录，避免残留"幽灵片段"被 AI 检索到
    delete_document_chunks(db_doc.id)

    # 同步删除物理文件，避免 uploads 目录无限膨胀
    if db_doc.file_path and os.path.exists(db_doc.file_path):
        os.remove(db_doc.file_path)

    db.delete(db_doc)
    db.commit()
    return {"message": "删除成功"}


# ---------- 文件上传 ----------

def _file_sha256(file_path: str) -> str:
    """计算文件内容的 SHA256 指纹（分块读取，大文件也不会一次性占满内存）。"""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(8192)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    force: bool = Form(False),
    db: Session = Depends(get_db),
):
    # 按扩展名判断文件类型（比浏览器报的 content_type 可靠，有的浏览器上传 .md 会报成 application/octet-stream）
    ext = os.path.splitext(file.filename or "")[1].lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise HTTPException(status_code=400, detail=f"暂不支持 {ext or '无后缀'} 文件，支持：txt/md/pdf/图片/docx/xlsx")

    # 用 UUID 命名物理文件，避免重名覆盖和路径穿越（title 存原始展示名）
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 算文件指纹并查重（放在解析前：重复文件直接拦截，不用白等 OCR）
    file_hash = _file_sha256(file_path)
    if not force:
        existing = db.query(models.Document).filter(models.Document.file_hash == file_hash).first()
        if existing:
            # 重复文件不入库，清理刚存盘的物理文件，返回 409 让前端弹确认框
            if os.path.exists(file_path):
                os.remove(file_path)
            return JSONResponse(status_code=409, content={
                "detail": f"文件内容与已有的《{existing.title}》完全相同",
                "duplicate": True,
                "existing_title": existing.title,
            })

    # 调用对应的提取器；解析失败返回 400 和具体原因（不再静默存空内容）
    try:
        content = extractor(file_path).strip()
    except Exception as e:
        # 解析失败时清理已保存的物理文件，避免磁盘残留
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"文件解析失败：{str(e)}")

    db_doc = models.Document(
        title=file.filename,
        content=content,
        file_path=file_path,
        file_hash=file_hash,
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # 分块并写入向量库（统一走 vector_store，与 update/delete 保持同一套逻辑）
    chunks = chunk_text(content)
    if chunks:
        add_document_chunks(db_doc.id, file.filename, content)

    # 提取不到文字时给出明确提示（可能是纯图片扫描件或空文件），方便用户判断是否真的入库
    if not chunks:
        return {
            "message": "上传成功，但未能提取到文字内容（可能是纯图片扫描件或空文件）",
            "filename": file.filename,
            "document_id": db_doc.id
        }
    return {
        "message": "上传成功",
        "filename": file.filename,
        "document_id": db_doc.id
    }


# ---------- AI 问答 ----------

@app.post("/chat")
def chat(req: schemas.ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    reply = run_chat(req.message, history)
    return {"reply": reply}


# 直接运行本文件时启动开发服务器（python main.py）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
