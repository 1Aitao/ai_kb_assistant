# 个人知识库 AI 助手

基于 FastAPI + Chroma + DeepSeek 实现的个人知识库问答系统。

支持上传 TXT、MD、PDF 文档，AI 会基于文档内容回答问题。

## 技术栈

- Python 3.12
- FastAPI
- SQLAlchemy + SQLite
- ChromaDB
- Sentence-Transformers
- DeepSeek API
- HTML + JavaScript 前端

## 已实现功能

- 文档增删改查（CRUD）
- 文件上传（.txt / .md / .pdf）
- 文档切片与向量存储
- 基于知识库的 AI 问答（RAG）
- 点击文档标题快速提问
- 前端 Web 页面交互

## 待实现功能

- 删除文档时同步删除向量
- 支持更多文件类型
- Docker 部署
- 用户认证

## 快速开始

### 1. 克隆项目

```bash
git clone https://gitee.com/1Aitao/ai_kb_assistant.git
cd ai_kb_assistant
