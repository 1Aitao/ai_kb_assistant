# 个人知识库 AI 助手（ai_kb_assistant）

一个基于 FastAPI + DeepSeek API + Chroma 向量数据库的多用户个人知识库问答系统。
支持上传 PDF / TXT / MD / DOCX / XLSX / 图片，AI 基于文档内容回答问题（流式输出）；同时可查询中国城市实时天气。

---

## 技术栈

- Python 3.12
- FastAPI（Web 框架）
- SQLAlchemy + SQLite（元数据/文档记录）
- Pydantic（接口数据校验）
- ChromaDB + Sentence-Transformers（向量数据库 + 嵌入模型，默认 `BAAI/bge-small-zh-v1.5`）
- jieba + rank_bm25（BM25 关键词检索，与向量检索做混合融合）
- LangChain + LangGraph（Agent 编排，ReAct 循环）
- DeepSeek API（大模型）
- 和风天气 API（天气工具）
- SSE（Server-Sent Events，流式输出）
- Uvicorn（ASGI 服务器）

---

## 已实现功能

- [x] 多用户系统：注册 / 登录，PBKDF2 密码哈希（200k 迭代 + 盐），token 鉴权
- [x] 用户数据隔离：文档、向量、聊天记录全链路按 `user_id` 过滤，互不可见；旧数据自动迁移
- [x] 对话历史持久化：聊天记录落库，刷新页面不丢
- [x] 文档的增删改查（CRUD）
- [x] 上传 PDF / TXT / MD / DOCX / XLSX / 图片，自动提取文字
- [x] 扫描件 OCR：逐页判断，仅对无文字层的页做 OCR；模型懒加载 + 启动预热，消除首次上传冷启动
- [x] 混合检索：向量检索（语义）+ BM25（关键词，jieba 中文分词）并行执行，RRF 加权融合（0.6 / 0.4）
- [x] 基于 Chroma 的 RAG 检索，支持按文档标题（`title`）精确过滤
- [x] LangGraph ReAct Agent：大模型自主决定调用工具（天气 / 列文档 / 查知识库）
- [x] 多城市天气查询（一次提问多个城市分别查询）
- [x] SSE 流式输出：回答逐块推送，打字机效果（fetch + ReadableStream）
- [x] 来源引用：回答知识库问题时标注 `来源：《文档名》第N段`，前端可点击查看原文
- [x] 知识边界兜底：知识库无相关内容时，可用通用知识回答，但明确标注「不来自知识库」
- [x] 前端 Markdown 渲染、点击文档快速提问

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

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| POST | `/register` | 无 | 注册账号 |
| POST | `/verify` | 无 | 登录校验，颁发 token |
| GET | `/documents` | token | 获取本人文档列表 |
| POST | `/documents` | token | 创建文档 |
| GET | `/documents/{doc_id}` | token | 获取单个文档 |
| GET | `/documents/{doc_id}/file` | token | 查看原文档 |
| PUT | `/documents/{doc_id}` | token | 修改文档（同步更新向量库） |
| DELETE | `/documents/{doc_id}` | token | 删除文档（同步清理向量库和物理文件） |
| POST | `/upload` | token | 上传文件（txt/md/pdf/docx/xlsx/图片，SHA256 查重） |
| POST | `/chat` | token | 聊天 / RAG 问答 / 天气查询（非流式） |
| POST | `/chat/stream` | token | 同上，SSE 流式输出 |
| GET | `/chat/history` | token | 获取本人聊天历史 |

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
├── models.py          # 数据库表结构（User / Document / ChatRecord）
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

### 提问回答流程（流式）

1. 用户提问，前端 fetch 请求 `/chat/stream`，不等待完整响应
2. LangGraph Agent 自主判断是否需要调用工具
3. 查知识库时，向量检索与 BM25 关键词检索并行执行，RRF 加权融合后取最相关片段
4. 模型生成的回答以 SSE 数据帧逐块推送到前端，打字机效果展示
5. 流式结束后，完整问答落库（`/chat/history` 可回溯）
6. 知识库无相关内容时，模型可用通用知识回答并标注「不来自知识库」

---

## 待完善（下一步）

- [ ] 来源引用前端卡片化展示
- [ ] token 持久化（重启服务不掉线）
- [ ] pytest 自动化测试
- [ ] Docker 部署 + 上线

---

## 作者

AItao & 涛哥
