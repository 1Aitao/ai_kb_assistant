"""路由层：HTTP 接口定义。业务逻辑下沉至 agent/vector_store/extractors 各层。"""
import os
import hashlib
import hmac
import secrets
import time
import uuid
from fastapi import FastAPI,File,Form,UploadFile,Depends,HTTPException,Header,Query
import json
import mimetypes
from datetime import datetime

from fastapi.responses import RedirectResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

import models
import schemas
from agent import run_chat, stream_chat
from config import settings
from database import engine, get_db, SessionLocal
from extractors import EXTRACTORS, warm_up_ocr
from vector_store import (
    add_chat_memory,
    add_document_chunks,
    clear_all_chat_memory,
    clear_all_document_chunks,
    delete_document_chunks,
    list_all_chat_memory_texts,
)

models.Base.metadata.create_all(bind=engine)

# 服务启动时预热 OCR 引擎，避免重启后第一次上传扫描件额外慢约 50%
warm_up_ocr()

# ---------- 用户隔离：密码哈希 + 老数据一次性迁移 ----------

_PBKDF2_ITERATIONS = 200_000  # 搅拌次数：越多越难暴力破解，登录慢一点点值得


def _hash_password(password: str) -> str:
    """密码搅碎机：生成随机盐（调料）+ PBKDF2 反复搅拌，库里只存"盐$碎渣"。"""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return f"{salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """验密码：用库里存的盐把输入再搅一次，比对碎渣是否一致（恒定时间比较防时序攻击）。"""
    try:
        salt, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(digest.hex(), digest_hex)


def _ensure_column(db: Session, table: str, column: str) -> None:
    """老库补列：create_all 只建新表，不会给已存在的表加新列，需手动 ALTER TABLE。"""
    cols = [row[1] for row in db.execute(text(f"PRAGMA table_info({table})")).fetchall()]
    if column not in cols:
        db.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} INTEGER"))
        db.commit()


def _migrate_legacy_data() -> None:
    """一次性迁移：老版本没有用户概念，全部旧数据划归管理员并补 user_id 标签。

    幂等：已有任何用户（迁移过/多用户安装）或没有任何旧数据时直接跳过。
    向量库老块的 metadata 没有 user_id 标签，只能全量清空重建（BM25 同步恢复）。
    """
    db = SessionLocal()
    try:
        _ensure_column(db, "documents", "user_id")
        _ensure_column(db, "chat_records", "user_id")
        if db.query(models.User).count() > 0:
            return
        has_docs = db.query(models.Document).count() > 0
        has_chats = db.query(models.ChatRecord).count() > 0
        if not (has_docs or has_chats):
            return
        admin = models.User(
            username=settings.ADMIN_USERNAME,
            password_hash=_hash_password(settings.ADMIN_PASSWORD),
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)

        # SQLite 旧记录补标签
        db.query(models.Document).update({models.Document.user_id: admin.id})
        db.query(models.ChatRecord).update({models.ChatRecord.user_id: admin.id})
        db.commit()

        # 向量库文档块：清空后按新标签重灌（BM25 索引随 add 自动恢复）
        clear_all_document_chunks()
        for doc in db.query(models.Document).all():
            add_document_chunks(doc.id, doc.title, doc.content, user_id=admin.id)

        # 聊天长期记忆：先备份原文 → 清空 → 带标签重灌
        texts = list_all_chat_memory_texts()
        clear_all_chat_memory()
        for t in texts:
            add_chat_memory(t, user_id=admin.id)

        print(
            f"[迁移] 旧数据已划归管理员 {settings.ADMIN_USERNAME}"
            f"（user_id={admin.id}），登录用户名/密码即管理员账号"
        )
    finally:
        db.close()


_migrate_legacy_data()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = settings.UPLOAD_DIR

# 通行证仓库：token → (它属于哪个 user_id, 过期时刻)（服务重启即清空，需要重新登录）。
# 工号对应关系只存在服务端这份登记本里，前端拿着 token 也伪造不出别人的身份。
# 每张票带 7 天保质期：过期作废并清理，登记本不再只增不减。
_valid_tokens: dict[str, tuple[int, float]] = {}
_TOKEN_EXPIRE_SECONDS = 7 * 24 * 3600

# 登录防爆破记账本：用户名 → (连续失败次数, 锁定截止时间戳)。
# 错 5 次锁 60 秒，锁定期内连哈希都不算——PBKDF2 20 万次迭代的开销不再被攻击者白嫖。
# 同样存内存（重启清零），本地 demo 够用。
_login_failures: dict[str, tuple[int, float]] = {}
_MAX_LOGIN_FAILURES = 5
_LOGIN_LOCK_SECONDS = 60


def _purge_expired_tokens() -> None:
    """清扫过期死票：借每次登录的时机全量扫一遍（票的量级小，开销可忽略）。

    之所以不给字典设数量上限：数量满了不知道该删谁——可能全是死票而活人被误删。
    保质期让每条记录自带"该不该留"的信息，到期的是死票，精确清除，活人不受伤。
    """
    now = time.time()
    expired = [t for t, (_uid, expires_at) in _valid_tokens.items() if now > expires_at]
    for t in expired:
        _valid_tokens.pop(t, None)


def check_access(
    x_access_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> int:
    """验票员：校验通行证并查出"你是谁"（user_id），交给后面的查询当暗号。

    支持两种带票方式：
    - 请求头 x-access-token：前端 fetch/XHR 用
    - 查询参数 ?token=xxx：iframe/img 标签的 src 无法自定义请求头，只能拼在 URL 上
    """
    token_str = x_access_token or token or ""
    entry = _valid_tokens.get(token_str)
    if entry is None:
        raise HTTPException(status_code=401, detail="未登录或通行证已过期")
    user_id, expires_at = entry
    if time.time() > expires_at:
        _valid_tokens.pop(token_str, None)  # 过期票当场作废并划掉，不占登记本
        raise HTTPException(status_code=401, detail="未登录或通行证已过期")
    return user_id


@app.get("/")
def home():
    # 登录页本身必须公开，否则浏览器连页面都加载不了，没法输口令
    return RedirectResponse(url="/static/index.html")


@app.post("/register")
def register(req: schemas.RegisterRequest, db: Session = Depends(get_db)):
    """注册新用户：用户名查重 + 密码立刻搅成哈希（明文不落库）。免票接口。"""
    username = req.username.strip()
    if not (3 <= len(username) <= 30):
        raise HTTPException(status_code=400, detail="用户名需 3~30 个字符")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    if db.query(models.User).filter(models.User.username == username).first():
        raise HTTPException(status_code=409, detail="用户名已被占用")
    user = models.User(username=username, password_hash=_hash_password(req.password))
    db.add(user)
    db.commit()
    return {"message": "注册成功，请登录", "username": username}


@app.post("/verify")
def verify(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    """验用户名+密码、发通行证（发票时在登记本写上工号）。前端登录页调用。免票接口。"""
    name = username.strip()

    # 防爆破检查：攒够 5 次失败后锁 60 秒，锁定期内直接 429，不查库不算哈希
    failures, locked_until = _login_failures.get(name, (0, 0.0))
    now = time.time()
    if failures >= _MAX_LOGIN_FAILURES:
        if now < locked_until:
            raise HTTPException(status_code=429, detail="失败次数过多，请 1 分钟后再试")
        _login_failures.pop(name, None)  # 刑满释放：锁过期后本次失败重新计数
        failures = 0

    user = db.query(models.User).filter(models.User.username == name).first()
    if user is None or not _verify_password(password, user.password_hash):
        # 用户不存在与密码错误返回同一句话，防止攻击者探测哪些用户名注册过；
        # 同时失败记账，攒够次数触发下一轮的锁定
        _login_failures[name] = (failures + 1, now + _LOGIN_LOCK_SECONDS)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    _login_failures.pop(name, None)  # 登录成功，销账
    _purge_expired_tokens()  # 借登录的时机清扫过期死票，登记本不会无限膨胀
    token = secrets.token_hex(32)
    _valid_tokens[token] = (user.id, time.time() + _TOKEN_EXPIRE_SECONDS)
    return {"token": token}


@app.post("/logout")
def logout(
    x_access_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
):
    """销票：把当前通行证从登记本划掉，服务端立即失效（不只是前端删 localStorage）。

    不做鉴权：销票至多作废"请求方已报出票号的票"——能报出票值的人本就能拿它访问，
    销掉不产生新的越权风险；销一张不存在的票也只是无害的空操作。
    """
    _valid_tokens.pop(x_access_token or token or "", None)
    return {"message": "已退出登录"}


# ---------- 文档 CRUD ----------

@app.get("/documents", response_model=list[schemas.DocumentResponse])
def get_documents(db: Session = Depends(get_db), user_id: int = Depends(check_access)):
    # 对暗号：只返回本用户的文档
    documents = db.query(models.Document).filter(models.Document.user_id == user_id).all()
    return documents


@app.post("/documents", response_model=schemas.DocumentResponse)
def create_document(doc: schemas.DocumentCreate, db: Session = Depends(get_db), user_id: int = Depends(check_access)):
    db_doc = models.Document(title=doc.title, content=doc.content, user_id=user_id)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


@app.get("/documents/{doc_id}", response_model=schemas.DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db), user_id: int = Depends(check_access)):
    # id 对上且 user_id 也对上才查得到，别人的文档在这里就是 404
    db_doc = (
        db.query(models.Document)
        .filter(models.Document.id == doc_id, models.Document.user_id == user_id)
        .first()
    )
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return db_doc


@app.get("/documents/{doc_id}/file")
def get_document_file(doc_id: int, db: Session = Depends(get_db), user_id: int = Depends(check_access)):
    """返回文档的原始物理文件：PDF/图片/文本浏览器直接预览，Office 文件触发下载。"""
    db_doc = (
        db.query(models.Document)
        .filter(models.Document.id == doc_id, models.Document.user_id == user_id)
        .first()
    )
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not db_doc.file_path or not os.path.exists(db_doc.file_path):
        raise HTTPException(status_code=404, detail="原始文件不存在（可能是手动创建的文档，或文件已被移动/删除）")

    ext = os.path.splitext(db_doc.file_path)[1].lower()
    if ext in (".txt", ".md"):
        media_type = "text/plain"
    else:
        media_type = mimetypes.guess_type(db_doc.file_path)[0] or "application/octet-stream"
    # 不传 filename 参数 → Content-Disposition 为 inline，浏览器对 PDF/图片直接内嵌预览而不是下载
    return FileResponse(db_doc.file_path, media_type=media_type)


@app.put("/documents/{doc_id}", response_model=schemas.DocumentResponse)
def update_document(doc_id: int, doc: schemas.DocumentCreate, db: Session = Depends(get_db), user_id: int = Depends(check_access)):
    db_doc = (
        db.query(models.Document)
        .filter(models.Document.id == doc_id, models.Document.user_id == user_id)
        .first()
    )
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db_doc.title = doc.title
    db_doc.content = doc.content
    db_doc.updated_at = datetime.now()
    db.commit()
    db.refresh(db_doc)

    # 同步向量库：先清除旧块，再按新内容重建，避免检索到过期内容
    delete_document_chunks(db_doc.id, user_id)
    add_document_chunks(db_doc.id, db_doc.title, db_doc.content, user_id=user_id)
    return db_doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db), user_id: int = Depends(check_access)):
    db_doc = (
        db.query(models.Document)
        .filter(models.Document.id == doc_id, models.Document.user_id == user_id)
        .first()
    )
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 同步向量库：先清向量块，再删数据库记录，避免残留"幽灵片段"被 AI 检索到
    delete_document_chunks(db_doc.id, user_id)

    # 同步删除物理文件，避免 uploads 目录无限膨胀
    if db_doc.file_path and os.path.exists(db_doc.file_path):
        os.remove(db_doc.file_path)

    db.delete(db_doc)
    db.commit()
    return {"message": "删除成功"}


# ---------- 文件上传 ----------

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 上传大小上限：20MB


def _save_upload(file_obj, dest_path: str) -> None:
    """把上传流写盘，边写边计数，超过上限立刻中止并清理半成品。

    必须边写边拦：若先收完再称重，超大文件已经灌满磁盘，闸门形同虚设。
    """
    written = 0
    try:
        with open(dest_path, "wb") as buffer:
            while True:
                block = file_obj.read(1024 * 1024)  # 每次搬 1MB
                if not block:
                    break  # 流读完了，正常收工
                written += len(block)
                if written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="文件超过 20MB 上传上限")
                buffer.write(block)
    except Exception:
        # 中止时把写了半截的文件删掉，不让磁盘残留垃圾
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


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
    user_id: int = Depends(check_access),
):
    # 按扩展名判断文件类型（比浏览器报的 content_type 可靠，有的浏览器上传 .md 会报成 application/octet-stream）
    ext = os.path.splitext(file.filename or "")[1].lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise HTTPException(status_code=400, detail=f"暂不支持 {ext or '无后缀'} 文件，支持：txt/md/pdf/图片/docx/xlsx")

    # 用 UUID 命名物理文件，避免重名覆盖和路径穿越（title 存原始展示名）
    # 物理文件也按用户分目录存放：uploads/{user_id}/xxx.ext
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, safe_name)
    # 边写边计数写盘，超过 20MB 上限当场拒收（413），防止大文件灌满磁盘
    _save_upload(file.file, file_path)

    # 算文件指纹并查重（放在解析前：重复文件直接拦截，不用白等 OCR；查重范围只限本用户）
    file_hash = _file_sha256(file_path)
    if not force:
        existing = (
            db.query(models.Document)
            .filter(models.Document.file_hash == file_hash, models.Document.user_id == user_id)
            .first()
        )
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
        user_id=user_id,
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # 分块并写入向量库（切分统一在 add_document_chunks 内部做，返回实际写入的块数。
    # 旧写法：这里先 chunk_text 判空、函数内部又切一遍——同一篇文档白切两次）
    chunk_count = add_document_chunks(db_doc.id, file.filename, content, user_id=user_id)

    # 提取不到文字时给出明确提示（可能是纯图片扫描件或空文件），方便用户判断是否真的入库
    if not chunk_count:
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
def chat(req: schemas.ChatRequest, user_id: int = Depends(check_access)):
    # 历史由后端记忆层管理，前端只传当前一句；user_id 由验票员从 token 查出
    reply = run_chat(req.message, user_id)
    return {"reply": reply}


@app.post("/chat/stream")
async def chat_stream(req: schemas.ChatRequest, user_id: int = Depends(check_access)):
    """流式问答（SSE）：AI 每生成一小段就推给前端，避免长时间白屏像卡死。

    前端用 fetch + ReadableStream 读取，格式：data: {"delta": "..."}\n\n，data: [DONE] 结束。
    """
    async def event_gen():
        try:
            async for piece in stream_chat(req.message, user_id):
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
        except Exception as e:  # 流中途出错也要给前端一个交代
            yield f"data: {json.dumps({'delta': f'\n\n出错了：{e}'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/history", response_model=list[schemas.ChatRecordResponse])
def get_chat_history(db: Session = Depends(get_db), user_id: int = Depends(check_access)):
    """返回本用户最近 N 轮聊天记录：倒序取最新若干条，再反转为时间正序给前端顺序渲染。"""
    limit = settings.CHAT_SHORT_TURNS * 2
    records = (
        db.query(models.ChatRecord)
        .filter(models.ChatRecord.user_id == user_id)
        .order_by(models.ChatRecord.id.desc())
        .limit(limit)
        .all()
    )
    records.reverse()
    return records


# 直接运行本文件时启动开发服务器（python main.py）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
