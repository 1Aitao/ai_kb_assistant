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
        print("geo_url:", geo_url)
        geo_response = requests.get(geo_url, timeout=10)
        print("geo status:", geo_response.status_code)
        print("geo text:", geo_response.text[:500])

        if geo_response.status_code != 200:
            return f"抱歉，天气服务暂时不可用（状态码：{geo_response.status_code}）。"

        try:
            geo_data = geo_response.json()
        except Exception:
            return "抱歉，天气服务返回格式异常。"

        if geo_data.get("code") != "200" or not geo_data.get("location"):
            return f"抱歉，找不到城市：{city}（错误码：{geo_data.get('code')}）。"

        city_id = geo_data["location"][0]["id"]
        print("city_id:", city_id)
        # 第二步：根据城市 ID 查实时天气

        weather_url = f"{QWEATHER_BASE_URL}/v7/weather/now?location={city_id}&key={api_key}"
        print("weather_url:", weather_url)
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


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定中国城市的实时天气情况，例如：北京、上海、深圳、广州",
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
    }
]

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
            # 扫描版/图片版 PDF 的 extract_text() 可能返回 None，需做空值兜底
            page_text = page.extract_text() or ""
            content += page_text + "\n"
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    # 去除首尾空白，如果提取结果为空则提示（不中断保存，但避免向量库入库空块）
    content = content.strip()

    db_doc = models.Document(title=file.filename, content=content)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # 把 content 切成每段 500 字（空内容则跳过入库，避免 Chroma 空文档报错）
    chunks = [content[i:i+500] for i in range(0, len(content), 500)] if content else []

    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        collection.add(
            documents=[chunk],
            metadatas=[{"title": file.filename, "chunk_index": idx, "document_id": db_doc.id}],
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
def chat(req: ChatRequest):
    # 第一步：从向量库检索相关文档
    context = ""
    try:
        results = collection.query(
            query_texts=[req.message],
            n_results=3
        )
        if results["documents"] and results["documents"][0]:
            context = "\n\n".join(results["documents"][0])
    except Exception:
        context = ""

    # 第二步：构建系统提示词（无论是否有 RAG 上下文，都必须声明天气工具的使用规则）
    weather_instruction = (
        "当用户询问任何城市的天气时，必须调用 get_weather 工具查询实时天气，"
        "不要基于常识回答。如果用户提到多个城市，请分别查询每个城市的天气。"
    )

    if context:
        system_content = (
            "你是一个基于知识库回答问题的 AI 助手。请优先参考以下文档内容回答问题。"
            "如果文档里没有相关信息，请基于你的知识回答用户问题。"
            f"\n\n{weather_instruction}"
            "\n\n参考文档：\n" + context
        )
    else:
        system_content = (
            f"你是一个有用的 AI 助手。{weather_instruction}"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": req.message}
    ]

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
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                continue  # 参数解析失败跳过，避免整轮崩溃

            if function_name == "get_weather":
                city = arguments.get("city", "")
                if not city:
                    continue
                weather_result = get_weather(city)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": weather_result
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
