PROMPT = """
你是数据分析专家，需要在已有代码基础上继续实现下一步。
已知数据摘要：
{summaries}
已有代码：
python
{base_code}

历史计划：
{plan}
当前要实现的计划：
{current_plan}
任务：
在已有代码基础上实现当前计划。
要求：
- 只返回一个 Python 代码块。
- 不要附加任何说明文字。
"""
