import asyncio
import os

from langchain_core.messages import HumanMessage
from langgraph_dsstar.langgraph_dsstar import LGConfig, LangGraphDSStar
# todo: 替换为你的OpenAI API密钥！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
os.environ["OPENAI_API_KEY"] = ""


async def main():
    config = LGConfig(preserve_artifacts=True, emit_steps=True, debug_attempts=5)
    agent = LangGraphDSStar(config)
    app = agent.build_graph()

    state = {
        "messages": [HumanMessage(content="请帮我分析这两个文件内容，在结果json文件中的final_answer的值不要生成json格式，直接生成方便用户阅读的字符串")],
        "data_files": ["/Users/azad/python project/wutong/data/数据表结构.xlsx", "/Users/azad/python project/wutong/data/测试题目.json"], #todo: 替换为你的数据文件路径
    }

    result_state = app.invoke(state)
    result = result_state.get('final_result')
    # 获得result中final_answer的值
    # async for step in app.astream(state):
    #     print(step)  # 可改成更精简的打印方式
    print("result: \n", result_state.get('final_result'))


if __name__ == "__main__":
    asyncio.run(main())
