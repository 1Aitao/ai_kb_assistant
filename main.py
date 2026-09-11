from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import shutil
import os
import cv2
import numpy as np

from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

import models
import schemas
from schemas import ChatRequest,ChatMessage
from database import engine, get_db

from fastapi.staticfiles import StaticFiles

import chromadb
from chromadb.utils import embedding_functions
import requests
import json

import urllib.parse

# 加载环境变量（.env 文件）
load_dotenv()
QWEATHER_KEY = os.getenv("QWEATHER_KEY")
QWEATHER_BASE_URL = os.getenv("QWEATHER_BASE_URL", "https://devapi.qweather.com")
# 城市查询（GeoAPI）使用独立域名，不能和天气接口混用
QWEATHER_GEO_BASE_URL = os.getenv("QWEATHER_GEO_BASE_URL", QWEATHER_BASE_URL)

def get_weather(city: str) -> str:
    """
    调用和风天气 API 查询指定城市天气。
    需要先注册和风天气:https://www.qweather.com/
    """
    if not QWEATHER_KEY:
        return "抱歉，天气服务未配置，请检查 QWEATHER_KEY。"
    # 清理 API Key（去除可能的换行符、空格）
    api_key = QWEATHER_KEY.strip()

    try:
        # 第一步：根据城市名查城市 ID（GeoAPI 使用独立域名 geoapi.qweather.com，路径 /v2/city/lookup）
        geo_url = (
            f"{QWEATHER_GEO_BASE_URL}/geo/v2/city/lookup"
            f"?location={urllib.parse.quote(city)}"
            f"&key={api_key}"
        )
        geo_response = requests.get(geo_url, timeout=10)

        if geo_response.status_code != 200:
            return f"抱歉，天气服务暂时不可用（状态码：{geo_response.status_code}）。"

        try:
            geo_data = geo_response.json()
        except Exception:
            return "抱歉，天气服务返回格式异常。"

        if geo_data.get("code") != "200" or not geo_data.get("location"):
            return f"抱歉，找不到城市：{city}（错误码：{geo_data.get('code')}）。"

        city_id = geo_data["location"][0]["id"]
        # 第二步：根据城市 ID 查实时天气
        weather_url = f"{QWEATHER_BASE_URL}/v7/weather/now?location={city_id}&key={api_key}"
        weather_response = requests.get(weather_url, timeout=10)

        if weather_response.status_code != 200:
            return "抱歉，天气查询失败（状态码非 200)。"

        try:
            weather_data = weather_response.json()
        except Exception:
            return "抱歉，天气服务返回格式异常。"

        if weather_data.get("code") != "200":
            return f"抱歉，天气查询失败（错误码：{weather_data.get('code')}）。"

        now = weather_data.get("now", {})
        text = now.get("text", "未知")
        temp = now.get("temp", "未知")
        wind_dir = now.get("windDir", "未知")
        return f"{city}当前天气：{text}，温度 {temp}℃，风向 {wind_dir}。"

    except requests.exceptions.Timeout:
        return "抱歉，天气服务请求超时。"
    except Exception as e:
        return f"抱歉，天气查询出错：{str(e)}"

def list_documents() -> str:
    """
    查询知识库中的所有文档标题列表
    """
    from database import SessionLocal
    db = SessionLocal()
    try:
        documents = db.query(models.Document).all()
        if not documents:
            return "知识库中没有任何文档。"

        titles = [f"{idx + 1}. {doc.title}" for idx, doc in enumerate(documents)]
        return "知识库中的文档：\n" + "\n".join(titles)
    finally:
        db.close()

def query_knowledge_base(query: str,title:str | None = None, n_results: int = 3) -> str:
    """
    基于 Chroma 向量库查询与问题相关的文档片段
    """
    try:
        where_filter = None
        if title:
            where_filter = {"title": title}

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        if not results["documents"] or not results["documents"][0]:
                return "没有找到相关内容。"

        chunks = results["documents"][0]
        return "相关内容：\n\n" + "\n\n".join(
            [f"[片段 {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]
        )
    except Exception as e:
        return f"查询知识库出错：{str(e)}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定中国城市的实时天气。当用户提到'天气'、'气温'、'下雨'、'几度'等词语时，必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "中国城市名称，例如：北京、上海、深圳"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "查询知识库中有哪些文档，当用户问'有什么文档'、'文档列表'、'知识库里有什么'时调用",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "查询知识库中与用户问题相关的文档片段。当用户提到'知识库'、'库里'、'文档'、'我上传的'、'关于XXX'等词语时，必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的问题或关键词，例如：学生管理系统怎么写"
                    },
                    "title": {
                        "type": "string",
                        "description": "可选，指定只在某个文档里检索时传入完整文档名（含后缀），例如：复习题(1)(1)(1).pdf"
                    }
                },
                "required": ["query"]
            }
        }
    }       
]

# 读取 API Key 和 Base URL
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 创建大模型客户端
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 初始化 Chroma 向量数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
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


@app.get("/")
def home():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

# 上传文件保存的文件夹
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- 向量库公共 Helper（上传/更新/删除共用，保证 SQLite 与 Chroma 一致） ----------

def chunk_text(content: str, chunk_size: int = 500) -> list[str]:
    """将文本切分为固定长度的块，用于向量库入库。

    Args:
        content: 原始文本
        chunk_size: 每块字符数，默认 500

    Returns:
        分块列表；content 为空时返回空列表
    """
    if not content:
        return []
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]


def add_document_chunks(doc_id: int, title: str, content: str) -> None:
    """将文档内容分块后写入 Chroma 向量库。"""
    for idx, chunk in enumerate(chunk_text(content)):
        if not chunk:
            continue
        collection.add(
            documents=[chunk],
            metadatas=[{"title": title, "chunk_index": idx, "document_id": doc_id}],
            ids=[f"{doc_id}_{idx}"],
        )


def delete_document_chunks(doc_id: int) -> None:
    """从 Chroma 向量库中移除指定文档的全部向量块。"""
    collection.delete(where={"document_id": doc_id})


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

    # 同步向量库：先清除旧块，再按新内容重建，避免检索到过期内容
    delete_document_chunks(db_doc.id)
    add_document_chunks(db_doc.id, db_doc.title, db_doc.content)
    return db_doc


# 删除文档
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    db_doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if db_doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 同步向量库：先清向量块，再删数据库记录，避免残留"幽灵片段"被 AI 检索到
    delete_document_chunks(db_doc.id)
    db.delete(db_doc)
    db.commit()
    return {"message": "删除成功"}


# ---------- 文件内容提取器（按扩展名分发，新增格式只需在 EXTRACTORS 里加一行） ----------

# 提取 txt/md 文字（直接读取文本文件）
def extract_txt_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# OCR 识别 PDF（扫描版/图片版 PDF 没有文字层，需要把每页渲染成图片后用 OCR 识别文字）
def ocr_pdf(file_path: str) -> str:
    import fitz  # pymupdf，负责把 PDF 页面渲染成图片
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        # 放大 2 倍渲染，提高小字识别率
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        # 直接在内存中传递图像字节，避免 Windows 上临时文件被占用导致删除失败
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        result, _ = ocr(img)
        if result:
            full_text += "\n".join(line[1] for line in result) + "\n"
    return full_text.strip()

# 提取 PDF 文字（优先读文字层，读不到再降级 OCR）
def extract_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = "".join((p.extract_text() or "") for p in reader.pages)
    if text.strip():
        return text.strip()
    return ocr_pdf(file_path)

# 提取图片文字（jpg/png 等直接 OCR 识别）
def extract_image(file_path: str) -> str:
    from rapidocr_onnxruntime import RapidOCR
    result, _ = RapidOCR()(file_path)
    if not result:
        return ""
    return "\n".join(line[1] for line in result).strip()

# 提取 Word 文档文字（只读段落，跳过空段落）
def extract_docx(file_path: str) -> str:
    from docx import Document as DocxDocument
    d = DocxDocument(file_path)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())

# 提取 Excel 表格内容（每个非空单元格用 | 拼接成一行）
def extract_xlsx(file_path: str) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(file_path, read_only=True, data_only=True)
    lines = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)

# 扩展名 → 提取函数 的路由表
EXTRACTORS = {
    ".txt": extract_txt_md, ".md": extract_txt_md,
    ".pdf": extract_pdf,
    ".png": extract_image, ".jpg": extract_image, ".jpeg": extract_image,
    ".docx": extract_docx,
    ".xlsx": extract_xlsx,
}


# 上传文件
@app.post("/upload")
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 按扩展名判断文件类型（比浏览器报的 content_type 可靠，有的浏览器上传 .md 会报成 application/octet-stream）
    ext = os.path.splitext(file.filename or "")[1].lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise HTTPException(status_code=400, detail=f"暂不支持 {ext or '无后缀'} 文件，支持：txt/md/pdf/图片/docx/xlsx")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 调用对应的提取器；解析失败返回 400 和具体原因（不再静默存空内容）
    try:
        content = extractor(file_path).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败：{str(e)}")

    db_doc = models.Document(title=file.filename, content=content)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # 分块并写入向量库（统一走 Helper，与 update/delete 保持同一套逻辑）
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


@app.post("/chat")
def chat(req: ChatRequest):

    system_instruction = (
        "你是一个严格遵守规则的 AI 助手。你的任务是判断用户意图，并调用相应工具。"
        "规则如下："
        "1. 用户询问任何城市的天气时，必须调用 get_weather 工具，禁止基于常识回答。"
        "2. 用户询问知识库里有哪些文档时，调用 list_documents 工具。"
        "3. 用户询问知识库中关于某个主题的内容时，调用 query_knowledge_base 工具，query 参数传用户想查的问题。"
        "   当用户明确提到文档名时，必须把 title 参数设为该文档的完整名称（含后缀），禁止混入其他文档。"
        "4. 如果用户提到多个城市，请分别查询每个城市的天气。"
        "5. 回答知识库相关问题时，必须只依据工具返回的内容；如果工具返回'没有找到相关内容'，就直接告知用户未找到，禁止自行编造答案。"
        "6. 对于与天气、知识库无关的问题，可以直接回答。"
    )


    messages = [{"role": "system", "content": system_instruction}]

    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": req.message})


    # 第三步：调用大模型，同时提供天气工具（包裹网络异常）
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,    # 【新增】支持多城市并行查
            temperature=0.1,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"大模型服务调用失败：{str(e)}")

    if not response or not response.choices:
        raise HTTPException(status_code=502, detail="大模型服务暂停，请稍后重试")
    choice = response.choices[0]

    # 第四步：如果大模型需要调用工具（天气查询）
    if choice.finish_reason == "tool_calls":
        messages.append(choice.message.model_dump())

        for tool_call in choice.message.tool_calls:
            function_name = tool_call.function.name
            result = None
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            if function_name == "get_weather":
                city = arguments.get("city", "")
                if city:
                    result = get_weather(city)

            elif function_name == "list_documents":
                result = list_documents()

            elif function_name == "query_knowledge_base":
                query = arguments.get("query", "")
                title = arguments.get("title")
                if query:
                    result = query_knowledge_base(query, title=title)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result if result else "工具执行失败或未知工具"
            })

        # 工具调用后再次请求大模型（同样包裹异常 + 空值校验）
        try:
            final_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"大模型二次调用失败：{str(e)}")

        if not final_response or not final_response.choices:
            raise HTTPException(status_code=502, detail="大模型二次响应为空，请稍后重试")

        return {"reply": final_response.choices[0].message.content}

    # 第五步：不需要工具，直接返回
    return {
        "reply": choice.message.content,
        "finish_reason": choice.finish_reason,
        "tool_calls": choice.message.tool_calls
    }


# 直接运行本文件时启动开发服务器（python main.py）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
