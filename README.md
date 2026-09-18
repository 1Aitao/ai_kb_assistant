# 个人知识库 AI 助手（ai_kb_assistant）

一个基于 FastAPI + DeepSeek API + Chroma 向量数据库的个人知识库问答系统。
支持上传 PDF / TXT / MD / DOCX / XLSX / 图片，AI 基于文档内容回答问题；同时可查询中国城市实时天气。

---

## 技术栈

- Python 3.12
- FastAPI（Web 框架）
- SQLAlchemy + SQLite（元数据/文档记录）
- Pydantic（接口数据校验）
- ChromaDB + Sentence-Transformers（向量数据库 + 嵌入模型，默认 `BAAI/bge-small-zh-v1.5`）
- LangChain + LangGraph（Agent 编排，ReAct 循环）
- DeepSeek API（大模型）
- 和风天气 API（天气工具）
- Uvicorn（ASGI 服务器）

---

## 已实现功能

- [x] 文档的增删改查（CRUD）
- [x] 上传 PDF / TXT / MD / DOCX / XLSX / 图片，自动提取文字
- [x] 扫描件 OCR：逐页判断，仅对无文字层的页做 OCR；模型懒加载 + 启动预热，消除首次上传冷启动
- [x] 基于 Chroma 的 RAG 检索，支持按文档标题（`title`）精确过滤
- [x] LangGraph ReAct Agent：大模型自主决定调用工具（天气 / 列文档 / 查知识库）
- [x] 多城市天气查询（一次提问多个城市分别查询）
- [x] 来源引用：回答知识库问题时标注 `来源：《文档名》第N段`
- [x] 知识边界兜底：知识库无相关内容时，可用通用知识回答，但明确标注「不来自知识库」
- [x] 前端 Markdown 渲染、点击文档快速提问、多轮对话历史

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

Windows 安装依赖：

```bash
venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
venv/bin/python -m pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入真实值：

```env
DEEPSEEK_API_KEY=你的_deepseek_api_key
QWEATHER_KEY=你的_qweather_api_key

# 以下为可选，不填则使用 config.py 中的默认值
# DEEPSEEK_BASE_URL=https://api.deepseek.com
# DEEPSEEK_MODEL=deepseek-chat
# LLM_TEMPERATURE=0.1
# QWEATHER_BASE_URL=https://devapi.qweather.com
# QWEATHER_GEO_BASE_URL=https://geoapi.qweather.com
# CHROMA_DIR=./chroma_db
# CHROMA_COLLECTION=documents
# EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
# CHUNK_SIZE=500
# CHUNK_OVERLAP=50
# SEARCH_RESULTS=3
# OCR_RENDER_SCALE=1.2
# UPLOAD_DIR=uploads
```

> `.env` 含真实密钥，已被 `.gitignore` 忽略，不会上传到 Git。

### 4. 启动服务

Windows：

```bash
venv\Scripts\python -m uvicorn main:app --reload
```

macOS / Linux：

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
| POST | `/upload` | 上传文件（txt/md/pdf/docx/xlsx/图片） |
| POST | `/chat` | 聊天 / RAG 问答 / 天气查询 |

自动生成的 API 文档：

```text
http://127.0.0.1:8000/docs
```

---

## 项目结构

```text
ai_kb_assistant/
├── main.py            # FastAPI 主入口，只保留路由
├── config.py          # 配置中心（集中管理 API Key / 路径 / 模型名等）
├── database.py        # SQLAlchemy 数据库连接配置
├── models.py          # 数据库表结构（Document）
├── schemas.py         # Pydantic 接口数据校验
├── vector_store.py    # Chroma 向量库封装（切分 / 入库 / 检索）
├── extractors.py      # 文档内容提取（按扩展名路由，含 OCR）
├── agent.py           # LangGraph Agent + 工具定义（天气 / 列文档 / 查知识库）
├── static/
│   └── index.html     # 前端页面
├── uploads/           # 上传文件存放目录
├── chroma_db/         # Chroma 向量库持久化目录
├── kb_app.db          # SQLite 数据库文件
├── .env               # 环境变量（不提交 Git）
├── .gitignore         # Git 忽略文件
├── requirements.txt   # 依赖清单
└── README.md          # 项目说明
```

---

## 核心流程说明

### 文档入库流程

1. `extractors.py` 按扩展名提取文字（PDF 有文字层直接读，扫描页走 OCR）
2. `vector_store.py` 的 `chunk_text` 把长文切分成块
3. 嵌入模型把每块文字转成向量
4. 向量与原文（含 `title` / `chunk_index` 元数据）一起存入 Chroma

### 提问回答流程

1. 用户提问，前端把问题 + 历史发给 `/chat`
2. LangGraph Agent 自主判断是否需要调用工具
3. 查知识库时，`query_knowledge_base` 检索相关片段并标注来源
4. 知识库无相关内容时，模型可用通用知识回答并标注「不来自知识库」
5. 最终回答返回前端，Markdown 渲染展示

---

## 待完善（下一步）

- [ ] 对话历史持久化（刷新不丢，落库）
- [ ] 来源引用前端卡片化展示
- [ ] pytest 自动化测试
- [ ] Docker 部署 + 上线

---

## 作者

AItao & 涛哥
