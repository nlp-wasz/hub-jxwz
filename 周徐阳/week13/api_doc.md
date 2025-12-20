# ChatBI/Stock Agent 接口文档（前后端交互）

> 默认后端启动命令：`python main_server.py`（FastAPI，端口 8000）  
> SSE 流式接口请用 `text/event-stream` 方式消费。

## 1. 通用
- `GET /v1/healthy`：健康检查（无返回体，空实现）。

## 2. 对话 Chat（SSE 输出）
- `POST /v1/chat/`：发起/续写对话（SSE 流）。  
  Body `RequestForChat`:  
  - `content` (str, 必填)：用户提问  
  - `user_name` (str, 必填)  
  - `session_id` (str, 选填)：会话 ID，用于上下文关联  
  - `task` (str, 选填)：`股票分析` / `数据BI` / 其他  
  - `tools` (List[str], 选填)：允许的 MCP 工具名；空则纯模型回答  
  - 其余可选：`image_content`/`file_content`/`url_content`/`audio_content`/`video_content`，`vison_mode`、`deepsearch_mode`、`sql_interpreter`、`code_interpreter`  
  返回：SSE 文本流；落库后会话消息入库。
- `POST /v1/chat/init`：初始化会话，返回 `{session_id}`。
- `POST /v1/chat/get`：查询会话消息列表。Query/Body: `session_id` (str)。
- `POST /v1/chat/delete`：删除会话及其消息。Query/Body: `session_id` (str)。
- `POST /v1/chat/list`：按用户列出会话。Query/Body: `user_name` (str)。
- `POST /v1/chat/feedback`：对消息点赞/点踩。Query/Body: `session_id` (str), `message_id` (int), `feedback` (bool)。

## 3. 用户 User
- `POST /v1/users/login`：登录。Body: `user_name`, `password`。
- `POST /v1/users/register`：注册。Body: `user_name`, `password`, `user_role`。
- `POST /v1/users/reset-password`：重置密码。Body: `user_name`, `password`(旧), `new_password`。
- `POST /v1/users/info`：查询用户信息。Query/Body: `user_name`。
- `POST /v1/users/reset-info`：修改角色/状态。Body: `user_name`, `user_role?`, `status?`。
- `POST /v1/users/delete`：删除用户。Query/Body: `user_name`。
- `POST /v1/users/list`：用户列表（分页未实现，固定前 200）。

## 4. 股票收藏 Stock（业务层）
- `POST /v1/stock/list_fav_stock`：查询用户自选。Query/Body: `user_name`。
- `POST /v1/stock/del_fav_stock`：删除自选。Query/Body: `user_name`, `stock_code`。
- `POST /v1/stock/add_fav_stock`：添加自选。Query/Body: `user_name`, `stock_code`。
- `POST /v1/stock/clear_fav_stock`：清空自选。Query/Body: `user_name`。

## 5. 数据 Data（占位，未实现）
- `POST /v1/data/download|create|upload|delete`：函数体未实现，仅占位。

## 6. 行情底层接口 AutoStock（挂载路径 `/stock`）
> 均为 `GET`，返回第三方 API 原始 JSON；参数为 Query。
- `/stock/get_stock_code` (`operation_id=get_stock_codes`)：股票列表，`keyword?` 支持模糊。
- `/stock/get_index_code`：指数列表。
- `/stock/get_industry_code`：行业/板块列表。
- `/stock/get_board_info`：大盘数据。
- `/stock/get_stock_rank`：股票排行，参数 `node`（市场/板块，必填）、`industryCode?`、`pageIndex`、`pageSize`、`sort`、`asc`。
- `/stock/get_month_line`：月 K 线，`code` 必填，`startDate?` `endDate?` `type`。
- `/stock/get_week_line`：周 K 线，同上。
- `/stock/get_day_line`：日 K 线，同上。
- `/stock/get_stock_info`：单只股票基础信息，`code` 必填。
- `/stock/get_stock_minute_data`：分时数据，`code` 必填。

## 7. MCP 服务（Agent 工具）
- MCP Server 端口：`8900`（`main_mcp.py` 使用 SSE transport）。工具来源：
  - AutoStock FastAPI 导入的行情工具（同上 `/stock/*`）  
  - `api/news.py`：新闻热点、GitHub 热榜、抖音热榜等  
  - `api/saying.py`：每日语录  
  - `api/tool.py`：天气、地址解析、手机号归属地、景点、花语、汇率换算  
 这些作为 Agent 的工具调用端点，不直接暴露给前端页面。

## 8. 统计
- 已实现前后端交互接口：对话 6 + 用户 7 + 股票收藏 4 + 行情接口 10 + 健康检查 1 = **28 个**（其中 Data 模块 4 个占位未实现，未计入）。  
- Agent 工具能力（MCP）：AutoStock 10 + News 5 + Saying 3 + Tool 6 = **24 个工具** 可被 Agent 调用。
