# 作业1:
复现现有职能助手代码，环境要有 fastmcp、streamlit、openai-agent 库。
python mcp_server_main.py
streamlit run streamlit_demo.py

![截屏2025-11-20 16.45.26.png](%E6%88%AA%E5%B1%8F2025-11-20%2016.45.26.png)


# 作业2:
尝试新定义一个工具，进行文本情感分析，输入文本判断文本的情感类别。最终可以在界面通过agent 在对话中调用这个工具。

**代码详见 4-项目案例-企业职能助手/mcp_server/sentiment.py**

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    pass

![截屏2025-11-20 17.18.59.png](%E6%88%AA%E5%B1%8F2025-11-20%2017.18.59.png)


# 作业3:
尝试需要在对话中选择工具，增加 tool_filter 的逻辑。
    - 查询新闻的时候，只调用news的工具
    - 调用工具的时候，只调用tools的工具

**代码详见 4-项目案例-企业职能助手/week12_3_streamlit.py**

![截屏2025-11-20 22.21.58.png](%E6%88%AA%E5%B1%8F2025-11-20%2022.21.58.png)
