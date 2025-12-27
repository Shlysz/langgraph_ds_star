import uvicorn
import json

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from sse_starlette.sse import EventSourceResponse

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from langgraph_dsstar.langgraph_dsstar import LGConfig, LangGraphDSStar

# todo: 替换为你的OpenAI API密钥
os.environ["OPENAI_API_KEY"] = ""

# --- 1. 初始化 FastAPI 与 Graph ---
app = FastAPI(title="SQL Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # 允许的来源。开发环境推荐用 ["*"] 允许所有 IP 和端口
    # 生产环境建议指定具体域名，如 ["http://localhost:5173"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有 Header
)

# 初始化持久化存储 (生产环境建议替换为 AsyncSqliteSaver 或 PostgresSaver)
checkpointer = MemorySaver()


# todo 定义图
config = LGConfig(preserve_artifacts=True, emit_steps=True, debug_attempts=5)
agent = LangGraphDSStar(config)
graph = agent.build_graph()


# --- 2. 定义请求/响应模型 ---
class QueryRequest(BaseModel):
    session_id: str = Field(..., description="会话ID，用于区分不同用户上下文")
    query: str = Field(..., description="用户的自然语言查询")


class QueryResponse(BaseModel):
    session_id: str
    generated_sql: Optional[str] = None
    sql_result: Optional[Any] = None
    human_answer: Optional[str] = None
    error: Optional[str] = None


@app.get("/chat/dsstar")
async def chat_stream(session_id: str, query: str, file_paths: list[str]):
    """
    SSE 接口：实时推送各个节点的产出结果
    """

    # 1. 准备输入
    config = {"configurable": {"thread_id": session_id}}
    input_state = {
        "messages": [SystemMessage(content="在结果json文件中的final_answer的值不要设置为json格式，直接生成方便用户阅读的字符串"),HumanMessage(content=query)],
        "data_files": file_paths
    }
    print(f"--- [SSE] New Stream Request: session_id={session_id}, query={query} ---")

    # 2. 定义生成器函数
    async def event_generator():
        # 使用 astream 监听图的更新
        # mode="values" 表示每次节点更新 state 后，推送最新的 state
        async for event in graph.astream(input_state, config=config, stream_mode="updates"):

            # event 的结构通常是: {'node_name': {'updated_field': 'value', ...}}

           if "execute_final" in event:
                data = event["execute_final"]
                if data.get("final_result"):
                    # 将str中的值提取为json格式
                    json_result = json.loads(data["final_result"])
                    print(f"✅ [SSE] Final Result: {data['final_result']}")
                    yield {
                        "event": "final_result",
                        "data": json_result.get['final_answer']
                    }

        yield {
            "event": "end",  # 事件名
            "data": "DONE"  # 数据内容（可以是任意字符串）
        }

    # 3. 返回 SSE 响应
    return EventSourceResponse(event_generator())


# --- 3. 核心接口 ---
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    # 构造 LangGraph 配置，指定 thread_id
    config = {"configurable": {"thread_id": request.session_id}}

    # 构造增量状态 (只传入新消息和当前查询)
    # LangGraph 会自动从 checkpointer 加载历史并合并
    input_state = {
        "messages": [HumanMessage(content=request.query)],
        "user_query": request.query,
    }

    try:
        # 使用异步调用 ainvoke 以支持高并发
        final_state = await graph.ainvoke(input_state, config=config)

        # 提取结果
        return QueryResponse(
            session_id=request.session_id,
            generated_sql=final_state.get("generated_sql"),
            sql_result=final_state.get("sql_result"),
            human_answer=final_state.get("human_answer"),
            error=final_state.get("sql_result_error") or final_state.get("human_answer_error")
        )

    except Exception as e:
        # 生产环境应记录日志 logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")


# --- 4. 启动入口 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12346)