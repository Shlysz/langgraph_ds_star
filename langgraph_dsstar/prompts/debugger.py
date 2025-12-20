PROMPT = """
已知可用文件（路径）：
{filenames}
已有代码：
python
{code}

错误信息：
{bug}
任务：
修复代码错误，并重新给出完整、可运行的 Python 脚本。
要求：
- 只返回一个 Python 代码块。
- 不要使用 try/except。
- 不要添加虚构数据。
"""
