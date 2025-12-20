PROMPT = """
你是数据分析专家，需要继续完善计划来回答问题。
问题：
{question}
已知数据摘要：
{summaries}
当前计划：
{plan}
当前步骤：
{current_step}
当前结果：
{result}
任务：
给出下一步计划，只写一句可执行的步骤，不要解释。
"""
