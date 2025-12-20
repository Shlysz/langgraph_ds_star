PROMPT = """
你是数据分析专家，需要基于参考代码与执行结果给出最终答案。
已知数据摘要：
{summaries}
参考代码：
python
{code}

执行结果：
{result}
问题：
{question}
输出规范：
{guidelines}
任务：
生成一段可运行的 Python 代码，输出符合规范的答案。
要求：
- 只返回一个 Python 代码块。
- 不要使用 try/except。
"""
