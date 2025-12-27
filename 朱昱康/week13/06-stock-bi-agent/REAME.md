

# 文件划分

- `main_mcp.py` - MCP Server 主程序 （模块1）
- `main_server.py` - Fast API Web服务主程序 （模块2）
- `REAME.md` - 项目说明文档（注意文件名拼写应为README.md）
- `./agent/` - 智能Agent模块（工作流、openai-agent代码）
  - `stock_agent.py` - 股票分析智能代理
  - `db_agent.py` - 数据库操作代理
  - `csv_agent.py` - CSV文件处理代理
  - `excel_agent.py` - Excel文件处理代理
  - `vison_agent.py` - 视觉分析代理
  - `summary_agent.py` - 摘要生成代理
- `./api/` - API接口模块（底层api、外部api）
  - `autostock.py` - 股票数据接口（付费接口）
  - `news.py` - 新闻数据接口
  - `saying.py` - 名言/语录接口
  - `tool.py` - 工具类接口
- `./models/` - 数据模型
  - `data_models.py` - Pydantic和typing数据模型定义
  - `orm.py` - 关系型数据库关系映射
- `./routers/` - FastAPI路由
  - `chat.py` - 聊天路由
  - `stock.py` - 股票数据路由
  - `user.py` - 用户管理路由
  - `data.py` - 数据管理路由
- `./services/` - 业务服务层
  - `chat.py` - 聊天服务
  - `stock.py` - 股票服务
  - `user.py` - 用户服务
- ./demo - Streamlit 前端页面 （模块3）
  - ./demo/chat/` - 聊天功能演示
    - `chat.py` - 聊天主程序
    - `chat_list.py` - 聊天列表管理
  - `./demo/stock/` - 股票功能演示
    - `stock_info.py` - 股票信息
    - `stock_kline.py` - K线图数据
    - `stock_search.py` - 股票搜索
    - `stock_rank.py` - 股票排名
    - `stock_board.py` - 板块数据
    - `stock_industry.py` - 行业数据
    - `stock_favorite.py` - 收藏功能
    - `stock_min_data.py` - 分钟数据
  - `./demo/user/` - 用户管理演示
    - `user_login.py` - 用户登录
    - `user_register.py` - 用户注册
    - `user_info.py` - 用户信息
    - `user_list.py` - 用户列表
    - `user_delete.py` - 用户删除
    - `user_reset.py` - 密码重置
  - `./demo/mcp/` - MCP协议演示
    - `mcp_list.py` - MCP列表
    - `mcp_debug.py` - MCP调试
  - `./demo/data/` - 数据管理演示
    - `data_list.py` - 数据列表
    - `data_manage.py` - 数据管理
- `./templates/` - 模板文件
  - `chat_start_system_prompt.jinjia2` - 聊天系统提示模板（Jinja2格式）
- `./assert/` - 资源文件
  - `conversations.db*` - 对话记录数据库
  - `sever.db` - 服务器数据库
  - `chat-message` - 聊天消息数据
- `./test/` - 单元测试文件
  - `test_agent.py` - 代理测试
  - `test_autostock.py` - 自动化股票测试
  - `test_jinja2_template.py` - 模板测试
  - `test_mcp.py` - MCP协议测试

# 开发过程
## 模块1: mcp 服务，2-3天

```code
mcp = FastMCP.from_fastapi(app=app) # 导入fastapi 服务

async def setup():
    await mcp.import_server(news_mcp, prefix="") # 定义mcp 服务
    await mcp.import_server(saying_mcp, prefix="")
    await mcp.import_server(tool_mcp, prefix="")
```

mcp server 非常灵活，可以来源比较多样；
- @mcp.tool： 在函数加注解，定义为mcp 服务 -> 通过mcp client调用
- 从fastapi导入（灵活）： 从fast api 导入为mcp 服务 -> 通过http、mcp client调用
- 从外部 openapi json 导入：导入外部其他的程序api json


## 模块2: Fast API 后端开发，3-4天

```code
# 前后端交互的http接口，与业务相关的api
# http://127.0.0.1:8000/docs#/
app.include_router(user_routers)
app.include_router(chat_routers)
app.include_router(data_routers)
app.include_router(stock_routers)

# 底层的股票http接口，底层相关的api
# http://127.0.0.1:8000/stock/docs#/
app.mount("/stock", stock_app)
```

- 底层数据库，models
- 业务代码，services
- fastapi路由，routers

## 模块3: Streamlit 前端开发，1-2天

> Streamlit 不是强制的，但是推荐掌握一下。帮助理解前后端如何交互。

## 开发过程
- 模块进行划分：用户管理模块、股票管理模块、对话模块（主要模块）
- 对话模块：
  - 历史对话记录，对话切换
  - 大模型历史状态管理的过程
    - AdvancedSQLiteSession：主要依赖openai-agent的服务，不是很透明，比较简单
    - 自己实现：历史对话也输入给大模型，基于历史输入继续对话（限制最后10条历史对话作为输入，如果对话太长大模型忘记最开始信息）
  - 对话：

