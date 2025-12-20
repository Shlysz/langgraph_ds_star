PROMPT = """
你是数据分析专家，需要判断当前计划与代码是否足以回答问题。
问题：
{question}
已知数据摘要：
{summaries}
计划：
{plan}
当前步骤：
{current_step}
代码：
python
{code}

代码执行结果：
{result}
任务：
判断是否已经足够回答问题。
只回答 Yes 或 No。
"""
