# Stock BI Agent API 文档

# Stock BI Agent API 文档

## 1. 用户管理

### 1.1 获取用户列表
**接口地址**: `http://localhost:8000/v1/users/list`  
**请求方式**: POST  
**接口描述**: 获取所有用户信息列表

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| 无 | 无需参数 | - | - |

**响应结果说明**
| 参数名称 | 参数说明 | 类型 |
|---------|---------|------|
| code | 状态码 | int |
| message | 响应消息 | string |
| data | 用户数据列表 | array |

**响应结果示例**:
```json
{
    "code": 200,
    "message": "ok",
    "data": [
        {
            "user_id": 1,
            "user_name": "admin",
            "user_role": "admin",
            "register_time": "2023-11-27T15:30:00",
            "status": true
        }
    ]
}
```

### 1.2 新增用户
**接口地址**: `http://localhost:8000/v1/users/register`  
**请求方式**: POST  
**接口描述**: 注册新用户

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| user_name | 用户名 | 是 | string |
| password | 密码 | 是 | string |
| user_role | 用户角色 | 是 | string |

**请求示例**:
```json
{
    "user_name": "testuser",
    "password": "password123",
    "user_role": "user"
}
```

**响应结果示例**:
```json
{
    "code": 200,
    "message": "用户注册成功",
    "data": []
}
```

## 2. 股票管理

### 2.1 获取用户收藏股票列表
**接口地址**: `http://localhost:8000/v1/stock/list_fav_stock`  
**请求方式**: POST  
**接口描述**: 获取用户收藏的股票列表

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| user_name | 用户名 | 是 | string |

**请求示例**:
```json
{
    "user_name": "testuser"
}
```

**响应结果示例**:
```json
{
    "code": 200,
    "message": "获取用户所有股票成功",
    "data": [
        {
            "stock_code": "000001.SZ",
            "create_time": "2023-11-27T15:30:00"
        }
    ]
}
```

### 2.2 新增收藏股票
**接口地址**: `http://localhost:8000/v1/stock/add_fav_stock`  
**请求方式**: POST  
**接口描述**: 添加股票到用户收藏

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| user_name | 用户名 | 是 | string |
| stock_code | 股票代码 | 是 | string |

**请求示例**:
```json
{
    "user_name": "testuser",
    "stock_code": "600000.SH"
}
```

**响应结果示例**:
```json
{
    "code": 200,
    "message": "添加成功",
    "data": []
}
```

## 3. 聊天功能

### 3.1 发送聊天消息
**接口地址**: `http://localhost:8000/v1/chat/`  
**请求方式**: POST  
**接口描述**: 发送聊天消息并获取AI回复

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| content | 消息内容 | 是 | string |
| user_name | 用户名 | 是 | string |
| session_id | 会话ID | 否 | string |
| task | 任务类型 | 否 | string |
| tools | 工具列表 | 否 | array |

**请求示例**:
```json
{
    "content": "你好",
    "user_name": "testuser",
    "session_id": "session_123456",
    "task": "stock_analysis",
    "tools": ["stock", "news"]
}
```

**响应结果说明**:
- 使用Server-Sent Events (SSE)流式返回
- 响应头: `Content-Type: text/event-stream`

### 3.2 获取聊天记录
**接口地址**: `http://localhost:8000/v1/chat/get`  
**请求方式**: POST  
**接口描述**: 获取指定会话的聊天记录

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| session_id | 会话ID | 是 | string |

**请求示例**:
```json
{
    "session_id": "session_123456"
}
```

**响应结果示例**:
```json
{
    "code": 200,
    "message": "ok",
    "data": [
        {
            "role": "user",
            "content": "你好",
            "timestamp": "2023-11-27T15:30:00"
        },
        {
            "role": "assistant",
            "content": "你好！我是AI助手，请问有什么可以帮您？",
            "timestamp": "2023-11-27T15:30:01"
        }
    ]
}
```

## 4. 股票数据接口

### 4.1 股票代码查询
**接口地址**: `http://localhost:8000/get_stock_code`  
**请求方式**: GET  
**接口描述**: 查询所有股票代码，支持模糊查询

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| keyword | 股票代码或名称关键字 | 否 | string |

**响应结果示例**:
```json
{
    "code": 0,
    "message": "success",
    "data": [
        {
            "code": "000001.SZ",
            "name": "平安银行"
        }
    ]
}
```

### 4.2 指数代码查询
**接口地址**: `http://localhost:8000/get_index_code`  
**请求方式**: GET  
**接口描述**: 查询所有指数代码

**响应结果示例**:
```json
{
    "code": 0,
    "message": "success",
    "data": [
        {
            "code": "000001.SH",
            "name": "上证指数"
        }
    ]
}
```

### 4.3 获取股票K线数据
**接口地址**: `http://localhost:8000/get_stock_day_kline`  
**请求方式**: GET  
**接口描述**: 获取股票的日K线数据

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| code | 股票代码 | 是 | string |
| startDate | 开始日期(YYYY-MM-DD) | 否 | string |
| endDate | 结束日期(YYYY-MM-DD) | 否 | string |
| type | 复权类型(0:不复权,1:前复权,2:后复权) | 否 | int |

**响应结果示例**:
```json
{
    "code": 0,
    "message": "success",
    "data": [
        {
            "date": "2023-11-27",
            "open": 10.5,
            "high": 10.8,
            "low": 10.4,
            "close": 10.7,
            "volume": 1000000
        }
    ]
}
```

### 4.4 获取股票分时数据
**接口地址**: `http://localhost:8000/get_stock_minute_data`  
**请求方式**: GET  
**接口描述**: 获取股票的分时交易数据

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| code | 股票代码 | 是 | string |

**响应结果示例**:
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "currentPrice": 10.5,
        "change": 0.2,
        "changePercent": 1.94,
        "volume": 1000000,
        "amount": 10500000,
        "time": "14:30:00"
    }
}
```

### 4.5 获取股票排行
**接口地址**: `http://localhost:8000/get_stock_rank`  
**请求方式**: GET  
**接口描述**: 获取股票涨跌幅排行

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| node | 市场代码(a:沪深A股) | 是 | string |
| industryCode | 行业代码 | 否 | string |
| pageIndex | 页码 | 否 | int |
| pageSize | 每页数量 | 否 | int |
| sort | 排序字段 | 否 | string |
| asc | 排序方式(0:降序,1:升序) | 否 | int |

**响应结果示例**:
```json
{
    "code": 0,
    "message": "success",
    "data": {
        "total": 100,
        "list": [
            {
                "code": "000001.SZ",
                "name": "平安银行",
                "price": 10.5,
                "changePercent": 3.45,
                "volume": 1000000,
                "amount": 10500000
            }
        ]
    }
}
```

## 5. 新闻与资讯

### 5.1 获取今日要闻
**接口地址**: `http://localhost:8000/v1/news/today`  
**请求方式**: GET  
**接口描述**: 获取今日重要新闻

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": [
        {
            "title": "今日重要新闻标题",
            "url": "https://example.com/news/1"
        }
    ]
}
```

### 5.2 获取抖音热点
**接口地址**: `http://localhost:8000/v1/news/douyin`  
**请求方式**: GET  
**接口描述**: 获取抖音热点话题

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": [
        {
            "title": "抖音热点话题",
            "hot": 1000000
        }
    ]
}
```

### 5.3 获取GitHub热门项目
**接口地址**: `http://localhost:8000/v1/news/github`  
**请求方式**: GET  
**接口描述**: 获取GitHub热门项目

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": [
        {
            "name": "项目名称",
            "url": "https://github.com/xxx",
            "stars": 1000
        }
    ]
}
```

### 5.4 获取体育新闻
**接口地址**: `http://localhost:8000/v1/news/sports`  
**请求方式**: GET  
**接口描述**: 获取最新体育新闻

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": [
        {
            "title": "体育新闻标题",
            "time": "2023-11-27 10:00:00"
        }
    ]
}
```

## 6. 名言警句

### 6.1 获取名人名言
**接口地址**: `http://localhost:8000/v1/saying/famous`  
**请求方式**: GET  
**接口描述**: 获取随机名人名言

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": "知识就是力量。"
}
```

### 6.2 获取励志名言
**接口地址**: `http://localhost:8000/v1/saying/motivation`  
**请求方式**: GET  
**接口描述**: 获取励志名言

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": "成功源于不懈的努力。"
}
```

### 6.3 获取工作鸡汤
**接口地址**: `http://localhost:8000/v1/saying/work`  
**请求方式**: GET  
**接口描述**: 获取工作相关的心灵鸡汤

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": "每天进步一点点，成功就在不远处。"
}
```

## 7. 数据管理

### 7.1 数据下载
**接口地址**: `http://localhost:8000/v1/data/download`  
**请求方式**: POST  
**接口描述**: 下载数据

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| file_id | 文件ID | 是 | string |
| user_id | 用户ID | 是 | string |

**响应结果**: 文件下载流

### 7.2 创建数据
**接口地址**: `http://localhost:8000/v1/data/create`  
**请求方式**: POST  
**接口描述**: 创建新数据

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| data | 数据内容 | 是 | object |
| user_id | 用户ID | 是 | string |

### 7.3 上传数据
**接口地址**: `http://localhost:8000/v1/data/upload`  
**请求方式**: POST  
**接口描述**: 上传文件数据

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| file | 上传的文件 | 是 | file |
| user_id | 用户ID | 是 | string |

### 7.4 删除数据
**接口地址**: `http://localhost:8000/v1/data/delete`  
**请求方式**: POST  
**接口描述**: 删除数据

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| data_id | 数据ID | 是 | string |
| user_id | 用户ID | 是 | string |

## 8. 通用响应状态码说明
**接口地址**: `http://localhost:8000/v1/news`  
**请求方式**: GET  
**接口描述**: 获取最新的财经新闻

**请求参数说明**
| 参数名称 | 参数说明 | 是否必须 | 类型 |
|---------|---------|---------|------|
| page | 页码 | 否 | int |
| pageSize | 每页数量 | 否 | int |

**响应结果示例**:
```json
{
    "code": 200,
    "message": "success",
    "data": [
        {
            "id": 1,
            "title": "央行降准0.5个百分点",
            "summary": "中国人民银行决定下调金融机构存款准备金率...",
            "publishTime": "2023-11-27T10:00:00"
        }
    ]
}
```

## 6. 通用响应状态码说明

| 状态码 | 说明 |
|-------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |
| 502 | 网关错误 |
| 503 | 服务不可用 |

## 5. 注意事项

1. 所有时间格式均为ISO 8601格式，例如：`"2023-11-27T15:30:00"`
2. 聊天接口(`/v1/chat/`)使用Server-Sent Events (SSE)进行流式响应
3. 需要身份验证的接口需要在请求头中添加`Authorization`字段
4. 分页参数说明：
   - pageNum: 当前页码，从1开始
   - pageSize: 每页显示条数，默认10条

## 6. 错误处理

当发生错误时，API会返回如下格式的响应：

```json
{
    "code": 400,
    "message": "错误描述信息",
    "data": null
}
```

常见错误信息：
- `用户不存在`
- `用户名或密码错误`
- `参数错误`
- `无权限访问`
- `系统异常，请联系管理员`
