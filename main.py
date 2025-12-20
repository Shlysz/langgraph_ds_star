import asyncio
import os

from langchain_core.messages import HumanMessage
from langgraph_dsstar.langgraph_dsstar import LGConfig, LangGraphDSStar
# todo: 替换为你的OpenAI API密钥！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
os.environ["OPENAI_API_KEY"] = "你的OpenAI API密钥"


async def main():
    config = LGConfig(preserve_artifacts=False, emit_steps=True)
    agent = LangGraphDSStar(config)
    app = agent.build_graph()

    state = {
        "messages": [HumanMessage(content="请分析文件的具体内容并总结结论")],
        "data_files": ["/data/sample_sales.csv"], #todo: 替换为你的数据文件路径
    }

    async for step in app.astream(state):
        print(step)  # 可改成更精简的打印方式


if __name__ == "__main__":
    asyncio.run(main())
