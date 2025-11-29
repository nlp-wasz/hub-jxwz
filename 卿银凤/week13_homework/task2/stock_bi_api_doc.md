# Stock BI Agent 接口文档

## 1. 项目与接口总览
- 服务类型：FastAPI 后端（`main_server.py`），前端通过 HTTP 与其交互。
- 统一响应：除 `/v1/chat/` 外其余业务接口均返回 `BasicResponse{code:int, message:str, data:any}`。
- 流式场景：`/v1/chat/` 通过 `StreamingResponse` 以 `text/event-stream` 推送增量内容。
- 总接口数：**32 个**（主服务 22 个，自带 Stock 工具 10 个）。
- 目录结构：用户、聊天、数据、股票 4 组业务路由 + `/stock` 下挂载的 Stock 工具 API。

| 模块 | 路由前缀 | 接口数 | 备注 |
| --- | --- | --- | --- |
| 健康检查 | `/v1/healthy` | 1 | 目前函数体为空，需返回值以避免 500 |
| 用户中心 | `/v1/users` | 7 | 基本 CRUD 与权限维护 |
| 对话服务 | `/v1/chat` | 6 | 包含 SSE 对话流与会话管理 |
| 数据管理 | `/v1/data` | 4 | 仅占位，未实现业务逻辑 |
| 自选股 | `/v1/stock` | 4 | 管理用户收藏股票 |
| Stock 工具 | `/stock/*` | 10 | 代理 autostock 第三方数据 |

## 2. 公共字段约定
- `code`：`200` 成功，`400` 失败，`404/500` 异常。
- `message`：提示。
- `data`：成功时返回业务数据结构，失败通常为空数组。
- 所有需要用户身份的接口均依赖 `user_name` 字段，不含 token 机制。

## 3. 接口详情

### 3.1 健康检查
| 方法 | 路径 | 请求体 | 响应 | 说明 |
| --- | --- | --- | --- | --- |
| GET | `/v1/healthy` | 无 | **当前未返回任何内容** | 需补充 `return {"status": "ok"}` 等以供前端使用 |

### 3.2 用户中心 `/v1/users`

| 方法 | 路径 | 请求体 / 参数 | 返回数据 | 描述 |
| --- | --- | --- | --- | --- |
| POST | `/login` | `RequestForUserLogin{user_name,password}` | `BasicResponse` | 校验账号密码 |
| POST | `/register` | `RequestForUserRegister{user_name,password,user_role}` | `BasicResponse` | 创建新用户；用户已存在返回 400 |
| POST | `/reset-password` | `RequestForUserResetPassword{user_name,password,new_password}` | `BasicResponse` | 先校验旧密码，再写入新密码 |
| POST | `/info` | `user_name`（query/form） | `BasicResponse(data=user_info)` | 获取单个用户信息 |
| POST | `/reset-info` | `RequestForUserChangeInfo{user_name, user_role?, status?}` | `BasicResponse` | 仅当字段存在才更新角色或状态 |
| POST | `/delete` | `user_name` | `BasicResponse` | 删除用户 |
| POST | `/list` | 无 | `BasicResponse(data=list[User])` | 用户列表 |

### 3.3 对话服务 `/v1/chat`

| 方法 | 路径 | 请求体 / 参数 | 返回数据 | 描述 |
| --- | --- | --- | --- | --- |
| POST | `/` | `RequestForChat`（包含 content、user_name、session_id、task、tools、可选多模态字段） | `StreamingResponse` SSE | 与 Agent 对话，前端需逐条解析 `data: ...\n\n` 片段 |
| POST | `/init` | 无 | `BasicResponse(data={"session_id": str})` | 生成新的会话 ID |
| POST | `/get` | `session_id` | `BasicResponse(data=session_detail)` | 查询指定会话历史 |
| POST | `/delete` | `session_id` | `BasicResponse` | 删除某会话 |
| POST | `/list` | `user_name` | `BasicResponse(data=list[ChatSession])` | 获取用户所有会话摘要 |
| POST | `/feedback` | `session_id`, `message_id`, `feedback:bool` | `BasicResponse` | 标记单条消息的反馈 |

### 3.4 数据管理 `/v1/data`

| 方法 | 路径 | 描述 |
| --- | --- | --- |
| POST | `/download` | 预留：下载数据集 |
| POST | `/create` | 预留：创建新数据集 |
| POST | `/upload` | 预留：上传文件 |
| POST | `/delete` | 预留：删除数据集 |

### 3.5 自选股 `/v1/stock`
| 方法 | 路径 | 请求参数 | 返回数据 | 描述 |
| --- | --- | --- | --- | --- |
| POST | `/list_fav_stock` | `user_name` | `BasicResponse(data=list[StockFavInfo])` | 获取用户收藏股票 |
| POST | `/add_fav_stock` | `user_name`, `stock_code` | `BasicResponse` | 添加收藏（内部去重） |
| POST | `/del_fav_stock` | `user_name`, `stock_code` | `BasicResponse` | 删除单个收藏 |
| POST | `/clear_fav_stock` | `user_name` | `BasicResponse` | 清空收藏 |

### 3.6 Stock 工具 `/stock/*`
> 该子应用通过 `app.mount("/stock", stock_app)` 暴露第三方数据，全部为 GET 请求，返回第三方 JSON，未包裹 `BasicResponse`。

| 路径 | 参数 | 描述 |
| --- | --- | --- |
| `/stock/get_stock_code` | `keyword?` | 查询全部股票，支持代码/名称模糊搜索 |
| `/stock/get_index_code` | 无 | 获取全部指数列表 |
| `/stock/get_industry_code` | 无 | 行业/板块排名数据 |
| `/stock/get_board_info` | 无 | 大盘行情 |
| `/stock/get_stock_rank` | `node, industryCode?, pageIndex, pageSize, sort, asc` | 股票排行，POST 请求体转发 |
| `/stock/get_month_line` | `code, startDate?, endDate?, type` | 月 K 线 |
| `/stock/get_week_line` | 同上 | 周 K 线 |
| `/stock/get_day_line` | 同上 | 日 K 线 |
| `/stock/get_stock_info` | `code` | 股票基础信息 |
| `/stock/get_stock_minute_data` | `code` | 分时行情 |


