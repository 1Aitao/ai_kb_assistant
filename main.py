from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from database import engine, get_db

# 创建所有数据表
models.Base.metadata.create_all(bind=engine)

# 创建 FastAPI 应用
app = FastAPI()

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

# 删除文档
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.delete(db_doc)
    db.commit()
    return {"message": "删除成功"}

@app.put("/documents/{doc_id}", response_model=schemas.DocumentResponse)
def update_document(doc_id: int, doc: schemas.DocumentCreate, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db_doc.title = doc.title
    db_doc.content = doc.content
    db.commit()
    db.refresh(db_doc)
    return db_doc

@app.put("/documents/{doc_id}", response_model=schemas.DocumentResponse)
def update_document(doc_id: int, doc: schemas.DocumentCreate, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db_doc.title = doc.title
    db_doc.content = doc.content
    db.commit()
    db.refresh(db_doc)
    return db_doc
