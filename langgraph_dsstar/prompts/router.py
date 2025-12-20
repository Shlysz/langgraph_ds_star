PROMPT = """
你是数据分析专家，当前计划不足以回答问题，需要决定下一步。
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
如果认为某一步是错误的，只能回答 Step 1, Step 2, ... Step K。
如果需要新增下一步，请回答 Add Step。
不要给出其它文字。
"""
