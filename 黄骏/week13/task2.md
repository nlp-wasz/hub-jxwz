# Stock BI Agent 前后端接口说明

## 1. 系统与模块概览

| 层级 | 说明 |
| --- | --- |
| 接口网关 | `main_server.py` 启动 FastAPI，挂载业务路由 `/v1/*` 及 AutoStock 透传应用 `/stock`。|
| 业务服务 | `routers/user.py`, `routers/chat.py`, `routers/stock.py`, `routers/data.py` 对应用户、对话、收藏股票与数据资产接口，服务逻辑位于 `services/*`。|
| 对外工具 | `api/autostock.py` 将 AutoStock HTTP API 暴露在 `/stock/...` 供前端与 Agent 调用；`api/news.py`, `api/saying.py`, `api/tool.py` 以 MCP 工具形式提供给智能体。|
| 数据存储 | SQLite (`assert/sever.db`) 通过 `models/orm.py` 定义用户、会话、收藏等表，另有 `assert/conversations.db` 保存流式 Agent 记忆。|
| 前端 | `demo/streamlit_demo.py` 管理导航，页面位于 `demo/{user,stock,data,chat,mcp}`，通过 HTTP 调用上述接口。|
| 智能体 | `services/chat.py` 结合 `agents` SDK、Qwen 模型与 MCP Server，提供流式对话与工具调用。|

所有业务接口均返回统一 `BasicResponse`：`{"code": int, "message": str, "data": Any}`（定义见 `models/data_models.py`）。`code=200` 表示成功，`400/404/500` 提示业务或异常。

---

## 2. 后端接口清单

### 2.1 通用约定

- **Base URL（默认本地开发）**：`http://127.0.0.1:8000`
- **鉴权**：当前无 Token，依赖前端 Session 管理。
- **Content-Type**：`application/json`（除 AutoStock 代理多为 GET）。
- **错误处理**：发生异常时返回 `BasicResponse(code=404/500, message=traceback, data=[])`。

### 2.2 用户服务 `routers/user.py`

| Method & Path | 功能 | 请求字段 | 返回 `data` | 前端入口 |
| --- | --- | --- | --- | --- |
| `POST /v1/users/login` | 校验用户名/密码（SHA256） | JSON `{user_name, password}` | 空列表 | `demo/user/user_login.py` |
| `POST /v1/users/register` | 创建用户 | JSON `{user_name, password, user_role}` | 空 | `demo/user/user_register.py` |
| `POST /v1/users/reset-password` | 校验旧密码并更新密码 | JSON `{user_name, password, new_password}` | 空 | （暂未在 Streamlit 暴露） |
| `POST /v1/users/info` | 查询用户信息 | Query `user_name` | `User` 模型 | `demo/user/user_info.py`, `user_list.py` |
| `POST /v1/users/reset-info` | 修改角色/状态 | JSON `{user_name, user_role?, status?}` | 空 | `demo/user/user_reset.py` |
| `POST /v1/users/delete` | 删除账号 | Query `user_name` | 空 | `demo/user/user_delete.py` |
| `POST /v1/users/list` | 分页列举用户（默认全部） | Query：可空 | `List[User]` | 仅管理员页面 `demo/user/user_list.py` |

### 2.3 对话服务 `routers/chat.py`

| Method & Path | 功能 | 请求字段 | 返回/流格式 | 前端入口 |
| --- | --- | --- | --- | --- |
| `POST /v1/chat/` | 流式联想问答，必要时调用 MCP 工具 | JSON `content`, `user_name`, `session_id?`, `task?`, `tools?`，其余富媒体字段留扩展 | `StreamingResponse` with `text/event-stream`，每个 chunk 为纯文本（或工具 JSON 段） | `demo/chat/chat.py` |
| `POST /v1/chat/init` | 预生成 `session_id` | 无 | `{"session_id": "..."}`
| `POST /v1/chat/get` | 获取某一会话内消息列表 | Query `session_id` | `List[{role, content, feedback,...}]` | `demo/chat/chat.py` 首次加载 |
| `POST /v1/chat/delete` | 删除整段会话 | Query `session_id` | 空 | `demo/chat/chat_list.py` |
| `POST /v1/chat/list` | 列举用户全部会话 | Query `user_name` | `List[ChatSession]` | `demo/chat/chat_list.py` |
| `POST /v1/chat/feedback` | 标记单条消息的好/差反馈 | Query `session_id`, `message_id`, `feedback` | 空 | （目前页面未接入） |

实现要点：

- `services/chat.py` 会按 `task` 选取系统提示（模板位于 `templates/chat_start_system_prompt.jinjia2`）。
- Agent 通过 `Runner.run_streamed` 将 OpenAI/Qwen 响应拆分为 `ResponseTextDeltaEvent`，逐块写回 SSE；调用 MCP 工具时会先输出 ```json``` 包裹的 `tool_name: {arguments}`，再续写 LLM 结论。
- Streamlit 前端将 `requests.post(..., stream=True)` 返回的 chunk 逐行渲染，若检测到工具调用（`get_day_line` 等）会再次命中 `/stock/...` 获取原始行情并绘图。

### 2.4 数据资产服务 `routers/data.py`

该路由准备了 `/v1/data/download|create|upload|delete` 四个 POST 入口，但函数体均为 `pass`，尚未实现。`demo/data/*` 目前亦为空文件；如需接入，可复用 `models/DataTable`。

### 2.5 收藏股票服务 `routers/stock.py`

| Method & Path | 功能 | 请求参数 | 返回 `data` | 前端入口 |
| --- | --- | --- | --- | --- |
| `POST /v1/stock/list_fav_stock` | 查询用户收藏 | Query `user_name` | `List[StockFavInfo{stock_code, create_time}]` | `demo/stock/stock_favorite.py` |
| `POST /v1/stock/add_fav_stock` | 添加收藏（去重） | Query `user_name`, `stock_code` | Bool | 同上 |
| `POST /v1/stock/del_fav_stock` | 删除单只股票 | Query `user_name`, `stock_code` | Bool | 同上 |
| `POST /v1/stock/clear_fav_stock` | 清空收藏 | Query `user_name` | Bool | 同上 |

底层使用 `services/stock.py` 操作 `user_favorite_stock` 表；若传入不存在的用户将返回空列表或 `True`（视操作而定），前端需结合登录状态约束。

### 2.6 AutoStock 代理接口（`app.mount("/stock", stock_app)`）

`api/autostock.py` 暴露的所有 GET 接口均直接透传 AutoStock SaaS，并返回原始 JSON。常用入口如下：

| Path | 说明 | 请求参数 | 典型使用页面 |
| --- | --- | --- | --- |
| `GET /stock/get_stock_code` | 股票模糊搜索 | `keyword?` | `demo/stock/stock_search.py` |
| `GET /stock/get_index_code` | 市场指数列表 | 无 | （暂未使用） |
| `GET /stock/get_industry_code` | 行业板块列表 | 无 | `demo/stock/stock_industry.py` |
| `GET /stock/get_board_info` | 大盘行情 | 无 | `demo/stock/stock_board.py` |
| `GET /stock/get_stock_rank` | 股票排行 | `node`、`industryCode?`、`pageIndex`、`pageSize`、`sort`、`asc` | `demo/stock/stock_rank.py` |
| `GET /stock/get_month_line` | 月 K 线 | `code`, `startDate?`, `endDate?`, `type` | `demo/stock/stock_kline.py`, Chat 工具 |
| `GET /stock/get_week_line` | 周 K 线 | 同上 | 同上 |
| `GET /stock/get_day_line` | 日 K 线 | 同上 | 同上 |
| `GET /stock/get_stock_info` | 单只股票详情（包含盘口、财务指标） | `code` | `demo/stock/stock_info.py` |
| `GET /stock/get_stock_minute_data` | 分时行情 | `code` | `demo/stock/stock_min_data.py` |

---

## 3. 前端（Streamlit）页面与接口映射

`demo/streamlit_demo.py` 根据 `st.session_state.logged` 切换菜单。下表列出主要页面与调用的后端：

| 页面 (`demo/...`) | 功能 | 依赖接口 |
| --- | --- | --- |
| `user/user_register.py` | 注册 | `POST /v1/users/register` |
| `user/user_login.py` | 登录/登出 | `POST /v1/users/login` |
| `user/user_info.py` | 查看当前用户信息 | `POST /v1/users/info` |
| `user/user_reset.py` | 修改角色/状态 | `POST /v1/users/reset-info` |
| `user/user_delete.py` | 注销账号 | `POST /v1/users/delete` |
| `user/user_list.py` | 管理员查看全量用户 | `POST /v1/users/info` (校验角色) + `POST /v1/users/list` |
| `stock/stock_search.py` | 股票/指数模糊检索 | `GET /stock/get_stock_code` |
| `stock/stock_board.py` | 大盘看板 | `GET /stock/get_board_info` |
| `stock/stock_industry.py` | 行业表 | `GET /stock/get_industry_code` |
| `stock/stock_rank.py` | 排行榜 + 分页 | `GET /stock/get_stock_rank` |
| `stock/stock_info.py` | 单股全量信息 | `GET /stock/get_stock_info` |
| `stock/stock_kline.py` | K 线可视化 | `/stock/get_{day,week,month}_line` |
| `stock/stock_min_data.py` | 分时价格/成交量 | `GET /stock/get_stock_minute_data` |
| `stock/stock_favorite.py` | 收藏 CRUD | `/v1/stock/list_fav_stock`, `add`, `del`, `clear` |
| `chat/chat.py` | 流式聊天 + 可选 MCP 工具 | `/v1/chat`, `/v1/chat/init`, `/v1/chat/get`, `/stock/get_*`（根据工具输出） |
| `chat/chat_list.py` | 会话历史 | `/v1/chat/list`, `/v1/chat/delete` |
| `mcp/mcp_list.py` | MCP 工具列表 | 直接访问 MCP Server `http://127.0.0.1:8900/sse` |
| `mcp/mcp_debug.py` | 调试单个 MCP 工具 | 同上 |
