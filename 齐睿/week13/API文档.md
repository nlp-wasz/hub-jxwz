# 股票BI智能助手 - API接口文档

**基础URL**: `http://127.0.0.1:8000`

**API文档**: `http://127.0.0.1:8000/docs` (Swagger UI)

---

## 目录

1. [用户管理接口](#用户管理接口)
2. [对话接口](#对话接口)
3. [股票收藏接口](#股票收藏接口)
4. [股票数据接口](#股票数据接口)
5. [MCP工具接口](#mcp工具接口)
6. [响应格式](#响应格式)
7. [错误处理](#错误处理)

---

## 用户管理接口

### 1. 用户登录

**端点**: `POST /v1/users/login`

**描述**: 用户登录认证

**请求体**:
```json
{
  "user_name": "admin",
  "password": "admin123"
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |
| password | string | ✓ | 密码 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "用户登陆成功",
  "data": []
}
```

**响应示例** (失败):
```json
{
  "code": 400,
  "message": "用户名或密码错误",
  "data": []
}
```

**状态码**:
- `200` - 登录成功
- `400` - 用户名或密码错误
- `404` - 服务器错误

---

### 2. 用户注册

**端点**: `POST /v1/users/register`

**描述**: 新用户注册

**请求体**:
```json
{
  "user_name": "newuser",
  "password": "password123",
  "user_role": "user"
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 (唯一) |
| password | string | ✓ | 密码 |
| user_role | string | ✓ | 用户角色 (user/admin) |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "用户注册成功",
  "data": []
}
```

**响应示例** (失败):
```json
{
  "code": 400,
  "message": "用户名已存在",
  "data": []
}
```

---

### 3. 重置密码

**端点**: `POST /v1/users/reset-password`

**描述**: 用户密码重置

**请求体**:
```json
{
  "user_name": "admin",
  "password": "oldpassword",
  "new_password": "newpassword123"
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |
| password | string | ✓ | 当前密码 |
| new_password | string | ✓ | 新密码 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "密码重置成功",
  "data": []
}
```

---

### 4. 获取用户信息

**端点**: `POST /v1/users/info`

**描述**: 获取指定用户的详细信息

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "获取用户信息成功",
  "data": {
    "user_id": 1,
    "user_name": "admin",
    "user_role": "admin",
    "register_time": "2025-01-01T10:00:00",
    "status": true
  }
}
```

---

### 5. 修改用户信息

**端点**: `POST /v1/users/reset-info`

**描述**: 修改用户角色或状态

**请求体**:
```json
{
  "user_name": "admin",
  "user_role": "admin",
  "status": true
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |
| user_role | string | ✗ | 用户角色 |
| status | boolean | ✗ | 用户状态 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "用户信息修改成功",
  "data": []
}
```

---

### 6. 删除用户

**端点**: `POST /v1/users/delete`

**描述**: 删除指定用户

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "用户删除成功",
  "data": []
}
```

---

### 7. 获取用户列表

**端点**: `POST /v1/users/list`

**描述**: 获取所有用户列表

**请求参数**: 无

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "user_id": 1,
      "user_name": "admin",
      "user_role": "admin",
      "register_time": "2025-01-01T10:00:00",
      "status": true
    },
    {
      "user_id": 2,
      "user_name": "user1",
      "user_role": "user",
      "register_time": "2025-01-02T10:00:00",
      "status": true
    }
  ]
}
```

---

## 对话接口

### 1. 初始化对话会话

**端点**: `POST /v1/chat/init`

**描述**: 创建新的对话会话

**请求参数**: 无

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "ok",
  "data": {
    "session_id": "a1b2c3d4e5f6"
  }
}
```

---

### 2. 流式对话

**端点**: `POST /v1/chat/`

**描述**: 发送对话请求并获取流式响应

**请求体**:
```json
{
  "content": "查询北京相关的股票信息",
  "user_name": "admin",
  "session_id": "a1b2c3d4e5f6",
  "task": "通用对话",
  "tools": ["get_stock_rank", "get_stock_code"]
}
```

**请求参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| content | string | ✓ | 用户提问内容 |
| user_name | string | ✓ | 用户名 |
| session_id | string | ✗ | 对话会话ID (新对话可不传) |
| task | string | ✗ | 对话任务 (通用对话/股票分析/数据BI) |
| tools | array | ✗ | 可用工具列表 |

**响应格式**: SSE流式响应 (text/event-stream)

**响应示例**:
```
北京相关的股票包括...
```

**工具列表** (可选):
- `get_stock_code` - 股票代码查询
- `get_stock_rank` - 股票排行
- `get_day_line` - 日K线数据
- `get_week_line` - 周K线数据
- `get_month_line` - 月K线数据
- `get_stock_info` - 股票信息
- `get_stock_minute_data` - 分时数据
- `get_today_daily_news` - 今日新闻
- `get_douyin_hot_news` - 抖音热点
- `get_city_weather` - 天气查询
- 等其他MCP工具

---

### 3. 获取对话历史

**端点**: `POST /v1/chat/get`

**描述**: 获取指定会话的对话历史

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| session_id | string | ✓ | 对话会话ID |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "id": 1,
      "role": "system",
      "content": "你好，我是AI助手...",
      "create_time": "2025-01-01T10:00:00",
      "feedback": null,
      "feedback_time": null
    },
    {
      "id": 2,
      "role": "user",
      "content": "查询北京相关的股票信息",
      "create_time": "2025-01-01T10:01:00",
      "feedback": null,
      "feedback_time": null
    },
    {
      "id": 3,
      "role": "assistant",
      "content": "北京相关的股票包括...",
      "create_time": "2025-01-01T10:02:00",
      "feedback": true,
      "feedback_time": "2025-01-01T10:03:00"
    }
  ]
}
```

**消息字段说明**:
| 字段 | 类型 | 说明 |
|------|------|------|
| id | integer | 消息ID |
| role | string | 消息角色 (system/user/assistant) |
| content | string | 消息内容 |
| create_time | datetime | 创建时间 |
| feedback | boolean | 反馈 (true/false/null) |
| feedback_time | datetime | 反馈时间 |

---

### 4. 删除对话会话

**端点**: `POST /v1/chat/delete`

**描述**: 删除指定的对话会话

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| session_id | string | ✓ | 对话会话ID |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "ok",
  "data": []
}
```

---

### 5. 获取用户对话列表

**端点**: `POST /v1/chat/list`

**描述**: 获取指定用户的所有对话会话

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "user_id": 1,
      "session_id": "a1b2c3d4e5f6",
      "title": "查询北京相关的股票信息",
      "start_time": "2025-01-01T10:00:00"
    },
    {
      "user_id": 1,
      "session_id": "f6e5d4c3b2a1",
      "title": "股票分析",
      "start_time": "2025-01-02T10:00:00"
    }
  ]
}
```

---

### 6. 对话反馈

**端点**: `POST /v1/chat/feedback`

**描述**: 对对话消息进行反馈 (点赞/点踩)

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| session_id | string | ✓ | 对话会话ID |
| message_id | integer | ✓ | 消息ID |
| feedback | boolean | ✓ | 反馈 (true=点赞, false=点踩) |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "ok",
  "data": []
}
```

---

## 股票收藏接口

### 1. 获取用户收藏股票列表

**端点**: `POST /v1/stock/list_fav_stock`

**描述**: 获取用户收藏的所有股票

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "获取用户所有股票成功",
  "data": [
    {
      "stock_code": "sh605589",
      "create_time": "2025-01-01T10:00:00"
    },
    {
      "stock_code": "sz000001",
      "create_time": "2025-01-02T10:00:00"
    }
  ]
}
```

---

### 2. 添加收藏股票

**端点**: `POST /v1/stock/add_fav_stock`

**描述**: 添加股票到收藏

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |
| stock_code | string | ✓ | 股票代码 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "添加成功",
  "data": []
}
```

---

### 3. 删除收藏股票

**端点**: `POST /v1/stock/del_fav_stock`

**描述**: 从收藏中删除股票

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |
| stock_code | string | ✓ | 股票代码 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "删除成功",
  "data": []
}
```

---

### 4. 清空收藏

**端点**: `POST /v1/stock/clear_fav_stock`

**描述**: 清空用户所有收藏

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| user_name | string | ✓ | 用户名 |

**响应示例** (成功):
```json
{
  "code": 200,
  "message": "删除成功",
  "data": []
}
```

---

## 股票数据接口

**基础URL**: `http://127.0.0.1:8000/stock`

**说明**: 这些接口既可以通过HTTP直接调用，也可以通过MCP工具调用

### 1. 获取股票代码

**端点**: `GET /stock/get_stock_code`

**描述**: 查询股票代码，支持模糊搜索

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| keyword | string | ✗ | 搜索关键词 (支持代码和名称) |

**请求示例**:
```
GET /stock/get_stock_code?keyword=北京
```

**响应示例** (成功):
```json
{
  "code": 200,
  "data": [
    {
      "code": "sh605589",
      "name": "圣泉集团"
    },
    {
      "code": "sz000001",
      "name": "平安银行"
    }
  ]
}
```

---

### 2. 获取股票排行

**端点**: `GET /stock/get_stock_rank`

**描述**: 获取股票排行榜

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| node | string | ✓ | 市场代码 (a/ash/asz/bsh/bsz) |
| industryCode | string | ✗ | 行业代码 |
| pageIndex | integer | ✗ | 页码 (默认1) |
| pageSize | integer | ✗ | 每页大小 (默认100) |
| sort | string | ✗ | 排序字段 (price/priceChange/pricePercent/volume/turnover) |
| asc | integer | ✗ | 排序方式 (0=降序, 1=升序) |

**请求示例**:
```
GET /stock/get_stock_rank?node=a&pageIndex=1&pageSize=10&sort=price&asc=0
```

**响应示例** (成功):
```json
{
  "code": 200,
  "data": [
    {
      "code": "sh605589",
      "name": "圣泉集团",
      "price": 25.50,
      "priceChange": 0.50,
      "pricePercent": 2.0,
      "volume": 1000000,
      "turnover": 25500000
    }
  ]
}
```

---

### 3. 获取日K线数据

**端点**: `GET /stock/get_day_line`

**描述**: 获取股票日K线数据

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| code | string | ✓ | 股票代码 |
| startDate | string | ✗ | 开始日期 (YYYY-MM-DD) |
| endDate | string | ✗ | 结束日期 (YYYY-MM-DD) |
| type | integer | ✗ | 复权类型 (0=不复权, 1=前复权, 2=后复权) |

**请求示例**:
```
GET /stock/get_day_line?code=sh605589&startDate=2025-01-01&endDate=2025-01-31&type=0
```

**响应示例** (成功):
```json
{
  "code": 200,
  "data": [
    ["2025-01-01", "24.50", "25.00", "25.50", "24.30", "1000000"],
    ["2025-01-02", "25.00", "25.20", "25.80", "24.80", "1200000"]
  ]
}
```

**数据格式**: [日期, 昨收, 今开, 最高, 最低, 成交量]

---

### 4. 获取周K线数据

**端点**: `GET /stock/get_week_line`

**描述**: 获取股票周K线数据

**请求参数**: 同日K线接口

**响应格式**: 同日K线接口

---

### 5. 获取月K线数据

**端点**: `GET /stock/get_month_line`

**描述**: 获取股票月K线数据

**请求参数**: 同日K线接口

**响应格式**: 同日K线接口

---

### 6. 获取股票基础信息

**端点**: `GET /stock/get_stock_info`

**描述**: 获取股票基础信息

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| code | string | ✓ | 股票代码 |

**请求示例**:
```
GET /stock/get_stock_info?code=sh605589
```

**响应示例** (成功):
```json
{
  "code": 200,
  "data": {
    "code": "sh605589",
    "name": "圣泉集团",
    "price": 25.50,
    "priceChange": 0.50,
    "pricePercent": 2.0,
    "open": 25.00,
    "high": 25.80,
    "low": 24.30,
    "volume": 1000000,
    "turnover": 25500000,
    "pe": 15.5,
    "pb": 2.3
  }
}
```

---

### 7. 获取分时数据

**端点**: `GET /stock/get_stock_minute_data`

**描述**: 获取股票当日分时数据

**请求参数** (Query):
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| code | string | ✓ | 股票代码 |

**请求示例**:
```
GET /stock/get_stock_minute_data?code=sh605589
```

**响应示例** (成功):
```json
{
  "code": 200,
  "data": [
    ["09:30", "25.00", "100000"],
    ["09:31", "25.05", "150000"],
    ["09:32", "25.10", "120000"]
  ]
}
```

**数据格式**: [时间, 价格, 成交量]

---

### 8. 获取行业代码

**端点**: `GET /stock/get_industry_code`

**描述**: 获取所有行业代码

**请求参数**: 无

**响应示例** (成功):
```json
{
  "code": 200,
  "data": [
    {
      "code": "01",
      "name": "农业"
    },
    {
      "code": "02",
      "name": "采矿业"
    }
  ]
}
```

---

### 9. 获取指数代码

**端点**: `GET /stock/get_index_code`

**描述**: 获取所有指数代码

**请求参数**: 无

**响应示例** (成功):
```json
{
  "code": 200,
  "data": [
    {
      "code": "sh000001",
      "name": "上证指数"
    },
    {
      "code": "sz399001",
      "name": "深证成指"
    }
  ]
}
```

---

### 10. 获取大盘数据

**端点**: `GET /stock/get_board_info`

**描述**: 获取市场大盘数据

**请求参数**: 无

**响应示例** (成功):
```json
{
  "code": 200,
  "data": {
    "upCount": 1500,
    "downCount": 1200,
    "flatCount": 300,
    "totalVolume": 50000000000,
    "totalTurnover": 500000000000
  }
}
```

---

## MCP工具接口

**说明**: 这些工具通过MCP协议调用，在对话中自动触发

### 新闻工具

| 工具名 | 描述 | 参数 |
|--------|------|------|
| get_today_daily_news | 获取今日新闻 | 无 |
| get_douyin_hot_news | 获取抖音热点 | 无 |
| get_github_hot_news | 获取GitHub趋势 | 无 |
| get_toutiao_hot_news | 获取头条新闻 | 无 |
| get_sports_news | 获取体育新闻 | 无 |

### 名言工具

| 工具名 | 描述 | 参数 |
|--------|------|------|
| get_today_familous_saying | 获取今日名言 | 无 |
| get_today_motivation_saying | 获取励志语录 | 无 |
| get_today_working_saying | 获取工作鸡汤 | 无 |

### 通用工具

| 工具名 | 描述 | 参数 |
|--------|------|------|
| get_city_weather | 获取城市天气 | city_name (城市拼音) |
| get_address_detail | 地址解析 | address_text (地址文本) |
| get_tel_info | 获取电话信息 | tel_no (电话号码) |
| get_scenic_info | 获取景点信息 | scenic_name (景点名称) |
| get_flower_info | 获取花语 | flower_name (花名) |
| get_rate_transform | 汇率转换 | source_coin, aim_coin, money |

---

## 响应格式

### 标准响应格式

所有API响应都遵循以下格式:

```json
{
  "code": 200,
  "message": "操作成功",
  "data": {}
}
```

**响应字段说明**:
| 字段 | 类型 | 说明 |
|------|------|------|
| code | integer | 状态码 (200=成功, 400=请求错误, 404=服务器错误) |
| message | string | 消息说明 |
| data | any | 响应数据 (可以是对象、数组或null) |

### 流式响应格式

对话接口使用SSE (Server-Sent Events) 流式响应:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no

北京相关的股票包括...
```

---

## 错误处理

### 常见错误码

| 错误码 | 说明 | 原因 |
|--------|------|------|
| 200 | 成功 | 请求成功 |
| 400 | 请求错误 | 参数错误、用户不存在等 |
| 404 | 服务器错误 | 内部异常、数据库错误等 |

### 错误响应示例

```json
{
  "code": 400,
  "message": "用户名或密码错误",
  "data": []
}
```

### 常见错误场景

1. **用户认证失败**
   - 错误码: 400
   - 原因: 用户名或密码错误
   - 解决: 检查用户名密码

2. **MCP连接失败**
   - 错误码: 404
   - 原因: MCP服务未启动
   - 解决: 启动main_mcp.py

3. **API超时**
   - 错误码: 404
   - 原因: 外部API响应超时
   - 解决: 重试或检查网络

4. **数据库错误**
   - 错误码: 404
   - 原因: 数据库连接失败
   - 解决: 检查数据库配置

---

## 使用示例

### Python示例

```python
import requests
import json

# 用户登录
response = requests.post(
    "http://127.0.0.1:8000/v1/users/login",
    json={"user_name": "admin", "password": "admin123"}
)
print(response.json())

# 初始化对话
response = requests.post("http://127.0.0.1:8000/v1/chat/init")
session_id = response.json()["data"]["session_id"]

# 发送对话请求
response = requests.post(
    "http://127.0.0.1:8000/v1/chat/",
    json={
        "content": "查询北京相关的股票信息",
        "user_name": "admin",
        "session_id": session_id,
        "tools": ["get_stock_rank", "get_stock_code"]
    },
    stream=True
)

# 处理流式响应
for chunk in response.iter_content(decode_unicode=True):
    if chunk:
        print(chunk, end="", flush=True)
```

### cURL示例

```bash
# 用户登录
curl -X POST "http://127.0.0.1:8000/v1/users/login" \
  -H "Content-Type: application/json" \
  -d '{"user_name":"admin","password":"admin123"}'

# 获取股票排行
curl -X GET "http://127.0.0.1:8000/stock/get_stock_rank?node=a&pageIndex=1&pageSize=10"

# 获取股票代码
curl -X GET "http://127.0.0.1:8000/stock/get_stock_code?keyword=北京"
```

---

## 速率限制

当前版本未实现速率限制，建议在生产环境中添加:

- 每个用户每分钟最多100个请求
- 每个IP每分钟最多1000个请求
- 大文件上传限制为100MB

---

## 安全建议

1. **认证**: 所有API调用都应包含用户认证
2. **HTTPS**: 生产环境应使用HTTPS
3. **API密钥**: 不要在代码中硬编码API密钥
4. **输入验证**: 验证所有用户输入
5. **日志**: 记录所有API调用用于审计

---

## 更新日志

### v1.0.0 (2025-01-27)
- 初始版本发布
- 实现用户管理、对话、股票收藏功能
- 集成MCP工具支持
- 支持流式对话响应
