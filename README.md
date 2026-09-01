# 个人知识库 AI 助手

一个基于 FastAPI + DeepSeek API + Chroma 向量数据库的个人知识库 AI 助手。

支持上传 PDF/TXT/MD 文档，AI 基于文档内容回答问题，同时可以查询中国城市的实时天气。

---

## 技术栈

- Python 3.12
- FastAPI
- SQLAlchemy + SQLite
- Pydantic
- ChromaDB + Sentence-Transformers
- DeepSeek API
- 和风天气 API
- Uvicorn

---

## 已实现功能

- [x] 文档的增删改查（CRUD）
- [x] 上传 PDF/TXT/MD 文件并提取文本
- [x] 基于 Chroma 向量数据库的 RAG 检索
- [x] 接入 DeepSeek API 的聊天对话
- [x] 接入和风天气 API，支持查询单个/多个城市天气
- [x] 前端页面展示与 Markdown 渲染

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/1Aitao/ai_kb_assistant.git
cd ai_kb_assistant
```

### 2. 创建虚拟环境

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
venv/bin/python -m pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件，内容如下：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
QWEATHER_KEY=your_qweather_key
QWEATHER_BASE_URL=https://devapi.qweather.com
QWEATHER_GEO_BASE_URL=https://geoapi.qweather.com
```

### 4. 启动服务

Windows:

```bash
venv\Scripts\python -m uvicorn main:app --reload
```

macOS/Linux:

```bash
venv/bin/python -m uvicorn main:app --reload
```

### 5. 打开网页

浏览器访问：

```text
http://127.0.0.1:8000/static/index.html
```

---

## API 接口说明

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/documents` | 获取所有文档 |
| POST | `/documents` | 创建文档 |
| GET | `/documents/{doc_id}` | 获取单个文档 |
| PUT | `/documents/{doc_id}` | 修改文档 |
| DELETE | `/documents/{doc_id}` | 删除文档 |
| POST | `/upload` | 上传文件（txt/md/pdf） |
| POST | `/chat` | 聊天/RAG/天气查询 |

自动生成 API 文档：

```text
http://127.0.0.1:8000/docs
```

---

## 项目结构

```text
ai_kb_assistant/
├── main.py           # FastAPI 主入口，包含所有路由和工具
├── database.py       # SQLAlchemy 数据库配置
├── models.py         # 数据库模型
├── schemas.py        # Pydantic 数据校验
├── static/
│   └── index.html    # 前端页面
├── uploads/          # 上传文件存放目录
├── chroma_db/        # Chroma 向量数据库持久化目录
├── kb_app.db         # SQLite 数据库文件
├── .env              # 环境变量（不上传 Git）
├── .gitignore        # Git 忽略文件
└── README.md         # 项目说明
```

---

## 截图

（待补充）

---

## 作者

AItao & 涛哥
