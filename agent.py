"""智能体层：LangChain @tool 定义工具 + LangGraph create_react_agent 接管工具调用循环。

LangChain 1.x 将 Agent 架构迁移至 LangGraph：
- 旧版 AgentExecutor / create_tool_calling_agent 已移除
- 新版 create_react_agent 内部构建图：LLM 节点 → 工具节点 → 循环，直到无工具调用
- prompt 参数直接传系统提示词字符串，无需 ChatPromptTemplate
"""
import logging
import urllib.parse

import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import models
from config import settings
from database import SessionLocal
from vector_store import search_chunks

logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """查询指定中国城市的实时天气。当用户提到天气、气温、下雨、几度等词语时调用。"""
    if not settings.QWEATHER_KEY:
        return "抱歉，天气服务未配置，请检查 QWEATHER_KEY。"
    api_key = settings.QWEATHER_KEY.strip()
    try:
        geo_url = (
            f"{settings.QWEATHER_GEO_BASE_URL}/geo/v2/city/lookup"
            f"?location={urllib.parse.quote(city)}&key={api_key}"
        )
        geo_resp = requests.get(geo_url, timeout=10)
        if geo_resp.status_code != 200:
            return f"抱歉，天气服务暂时不可用（状态码：{geo_resp.status_code}）。"
        geo_data = geo_resp.json()
        if geo_data.get("code") != "200" or not geo_data.get("location"):
            return f"抱歉，找不到城市：{city}（错误码：{geo_data.get('code')}）。"
        city_id = geo_data["location"][0]["id"]
        weather_resp = requests.get(
            f"{settings.QWEATHER_BASE_URL}/v7/weather/now?location={city_id}&key={api_key}",
            timeout=10,
        )
        if weather_resp.status_code != 200:
            return "抱歉，天气查询失败（状态码非 200）。"
        weather_data = weather_resp.json()
        if weather_data.get("code") != "200":
            return f"抱歉，天气查询失败（错误码：{weather_data.get('code')}）。"
        now = weather_data.get("now", {})
        return (
            f"{city}当前天气：{now.get('text', '未知')}，"
            f"温度 {now.get('temp', '未知')}℃，风向 {now.get('windDir', '未知')}。"
        )
    except requests.exceptions.Timeout:
        return "抱歉，天气服务请求超时。"
    except Exception as e:
        return f"抱歉，天气查询出错：{str(e)}"


@tool
def list_documents() -> str:
    """查询知识库中有哪些文档。当用户问有什么文档、文档列表、知识库里有什么时调用。"""
    db = SessionLocal()
    try:
        documents = db.query(models.Document).all()
        if not documents:
            return "知识库中没有任何文档。"
        titles = [f"{idx + 1}. {doc.title}" for idx, doc in enumerate(documents)]
        return "知识库中的文档：\n" + "\n".join(titles)
    finally:
        db.close()


@tool
def query_knowledge_base(query: str, title: str | None = None) -> str:
    """查询知识库中与问题相关的文档片段。当用户询问知识库中某个主题的内容时调用。

    Args:
        query: 要检索的问题或关键词
        title: 可选，指定只在某个文档内检索时传完整文档名（含后缀）
    """
    try:
        chunks = search_chunks(query, title=title)
        if not chunks:
            return "没有找到相关内容。"
        return "相关内容：\n\n" + "\n\n".join(
            f"[片段 {i + 1}]（来源：《{chunk['title']}》）\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        )
    except Exception as e:
        return f"查询知识库出错：{str(e)}"


SYSTEM_PROMPT = (
    "你是一个严格遵守规则的 AI 助手。你的任务是判断用户意图，并调用相应工具。"
    "规则如下："
    "1. 用户询问任何城市的天气时，必须调用 get_weather 工具，禁止基于常识回答。"
    "2. 用户询问知识库里有哪些文档时，调用 list_documents 工具。"
    "3. 用户询问知识库中关于某个主题的内容时，调用 query_knowledge_base 工具，"
    "query 参数传用户想查的问题。当用户明确提到文档名时，必须把 title 参数设为"
    "该文档的完整名称（含后缀），禁止混入其他文档。"
    "4. 如果用户提到多个城市，请分别查询每个城市的天气。"
    "5. 回答知识库相关问题时，优先依据工具返回的内容，并标注来源。"
    "如果工具返回'没有找到相关内容'，可以用你自己的通用知识回答，"
    "但必须明确说明'以下内容不来自知识库'，让用户能区分。"
    "6. 对于与天气、知识库无关的问题，可以直接回答。"
    "7. 回答知识库相关问题时，在末尾另起一行标注来源文档，"
    "格式：来源：《文档名》。如果来自多篇文档，全部列出。"
)

llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=settings.LLM_TEMPERATURE,
    parallel_tool_calls=True,
)

_tools = [get_weather, list_documents, query_knowledge_base]

# LangGraph ReAct Agent：内部构建 LLM → 工具 → 循环图，替代旧版 AgentExecutor
agent_executor = create_react_agent(
    model=llm,
    tools=_tools,
    prompt=SYSTEM_PROMPT,
)


def run_chat(message: str, history: list[dict]) -> str:
    """执行一轮对话：自动完成工具调用与最终回答。

    Args:
        message: 用户本轮输入
        history: 历史消息，格式 [{"role": "...", "content": "..."}]

    Returns:
        助手最终回复文本
    """
    # 将历史消息转为 LangChain 消息对象（系统提示由 agent 内部注入，无需手动添加）
    msgs = []
    for h in history:
        if h["role"] == "user":
            msgs.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            msgs.append(AIMessage(content=h["content"]))
    msgs.append(HumanMessage(content=message))

    try:
        result = agent_executor.invoke({"messages": msgs})
        # 从消息列表末尾找到最后一条有内容的 AI 消息作为最终回复
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage) and m.content:
                return m.content
        return ""
    except Exception as e:
        logger.exception("Agent 执行失败")
        return f"抱歉，AI 服务暂时不可用：{str(e)}"
