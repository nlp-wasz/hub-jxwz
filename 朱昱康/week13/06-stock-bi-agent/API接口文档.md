## 概述

股票分析 Agent 系统的前后端接口规范，包括用户管理、股票数据、智能对话和数据管理等模块的 API 接口。

## 基础信息

- **基础 URL**: `http://127.0.0.1:8000`
- **认证方式**: 基于用户名和密码的会话认证
- **数据格式**: JSON

## 通用响应格式

所有 API 接口都遵循统一的响应格式：

```json
{
  "code": 200, // 状态码：200成功，400客户端错误，404服务端错误
  "message": "ok", // 响应消息
  "data": [] // 响应数据，可能是对象或数组
}
```

## 1. 用户管理模块

### 1.1 用户登录

- **接口路径**: `/v1/users/login`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string", // 用户名
  "password": "string" // 密码
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "用户登陆成功",
  "data": []
}
```

### 1.2 用户注册

- **接口路径**: `/v1/users/register`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string", // 用户名
  "password": "string", // 密码
  "user_role": "string" // 用户角色
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "用户注册成功",
  "data": []
}
```

### 1.3 重置密码

- **接口路径**: `/v1/users/reset-password`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string", // 用户名
  "password": "string", // 原密码
  "new_password": "string" // 新密码
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "密码重置成功",
  "data": []
}
```

### 1.4 获取用户信息

- **接口路径**: `/v1/users/info`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string" // 用户名
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "获取用户信息成功",
  "data": {
    "user_id": 1,
    "user_name": "admin",
    "user_role": "admin",
    "register_time": "2023-01-01T00:00:00",
    "status": true
  }
}
```

### 1.5 修改用户信息

- **接口路径**: `/v1/users/reset-info`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string", // 用户名
  "user_role": "string", // 用户角色(可选)
  "status": true // 用户状态(可选)
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "用户信息修改成功",
  "data": []
}
```

### 1.6 删除用户

- **接口路径**: `/v1/users/delete`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string" // 用户名
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "用户删除成功",
  "data": []
}
```

### 1.7 获取用户列表

- **接口路径**: `/v1/users/list`
- **请求方法**: POST
- **请求参数**: 无

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "user_id": 1,
      "user_name": "admin",
      "user_role": "admin",
      "register_time": "2023-01-01T00:00:00",
      "status": true
    }
  ]
}
```

## 2. 智能对话模块

### 2.1 发起对话

- **接口路径**: `/v1/chat/`
- **请求方法**: POST
- **请求参数**:

```json
{
  "content": "string", // 用户提问内容
  "user_name": "string", // 用户名
  "session_id": "string", // 对话会话ID(可选)
  "task": "string", // 对话任务类型(可选)
  "tools": ["string"], // 选择的工具列表(可选)
  "image_content": "string", // 图片内容(可选)
  "file_content": "string", // 文件内容(可选)
  "url_content": "string", // URL内容(可选)
  "audio_content": "string", // 音频内容(可选)
  "video_content": "string", // 视频内容(可选)
  "vison_mode": true, // 视觉模式(可选)
  "deepsearch_mode": true, // 深度搜索模式(可选)
  "sql_interpreter": true, // SQL解释器(可选)
  "code_interpreter": true // 代码解释器(可选)
}
```

- **响应**: 流式响应(Server-Sent Events)，实时返回对话内容

### 2.2 初始化对话

- **接口路径**: `/v1/chat/init`
- **请求方法**: POST
- **请求参数**: 无

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": {
    "session_id": "abc123def456" // 新生成的会话ID
  }
}
```

### 2.3 获取对话记录

- **接口路径**: `/v1/chat/get`
- **请求方法**: POST
- **请求参数**:

```json
{
  "session_id": "string" // 会话ID
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "id": 1,
      "create_time": "2023-01-01T00:00:00",
      "feedback": true,
      "feedback_time": "2023-01-01T00:00:00",
      "role": "user",
      "content": "用户消息内容"
    },
    {
      "id": 2,
      "create_time": "2023-01-01T00:00:00",
      "feedback": null,
      "feedback_time": null,
      "role": "assistant",
      "content": "助手回复内容"
    }
  ]
}
```

### 2.4 删除对话

- **接口路径**: `/v1/chat/delete`
- **请求方法**: POST
- **请求参数**:

```json
{
  "session_id": "string" // 会话ID
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": []
}
```

### 2.5 获取对话列表

- **接口路径**: `/v1/chat/list`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string" // 用户名
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "user_id": 1,
      "session_id": "abc123def456",
      "title": "对话标题",
      "start_time": "2023-01-01T00:00:00",
      "feedback": true,
      "feedback_time": "2023-01-01T00:00:00"
    }
  ]
}
```

### 2.6 对话反馈

- **接口路径**: `/v1/chat/feedback`
- **请求方法**: POST
- **请求参数**:

```json
{
  "session_id": "string", // 会话ID
  "message_id": 1, // 消息ID
  "feedback": true // 反馈状态
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": []
}
```

## 3. 股票数据模块

### 3.1 获取用户收藏股票列表

- **接口路径**: `/v1/stock/list_fav_stock`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string" // 用户名
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "获取用户所有股票成功",
  "data": [
    {
      "stock_code": "sh600519",
      "create_time": "2023-01-01T00:00:00"
    }
  ]
}
```

### 3.2 添加收藏股票

- **接口路径**: `/v1/stock/add_fav_stock`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string", // 用户名
  "stock_code": "string" // 股票代码
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "添加成功",
  "data": []
}
```

### 3.3 删除收藏股票

- **接口路径**: `/v1/stock/del_fav_stock`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string", // 用户名
  "stock_code": "string" // 股票代码
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "删除成功",
  "data": []
}
```

### 3.4 清空收藏股票

- **接口路径**: `/v1/stock/clear_fav_stock`
- **请求方法**: POST
- **请求参数**:

```json
{
  "user_name": "string" // 用户名
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "删除成功",
  "data": []
}
```

## 4. 股票底层 API 模块

### 4.1 获取股票代码列表

- **接口路径**: `/stock/get_stock_code`
- **请求方法**: GET
- **请求参数**:

```
keyword: string (可选)  // 支持代码和名称模糊查询
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "code": "sh600519",
      "name": "贵州茅台"
    }
  ]
}
```

### 4.2 获取股票基础信息

- **接口路径**: `/stock/get_stock_info`
- **请求方法**: GET
- **请求参数**:

```
code: string  // 股票代码
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "code": "sh600519",
      "name": "贵州茅台",
      "price": "1800.00",
      "priceChange": "10.00",
      "changePercent": "0.56",
      "open": "1790.00",
      "close": "1790.00",
      "high": "1810.00",
      "low": "1785.00",
      "volume": "10000",
      "turnover": "18000000",
      "turnoverRate": "0.5",
      "pe": "30.5",
      "pb": "10.2",
      "totalWorth": "2250000000000"
    }
  ]
}
```

### 4.3 获取日 K 线数据

- **接口路径**: `/stock/get_day_line`
- **请求方法**: GET
- **请求参数**:

```
code: string           // 股票代码
startDate: string      // 开始时间(可选)
endDate: string        // 结束时间(可选)
type: int              // 0不复权,1前复权,2后复权
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    ["2023-01-01", "1790.00", "1795.00", "1810.00", "1785.00", "10000"],
    ["2023-01-02", "1800.00", "1805.00", "1820.00", "1795.00", "12000"]
  ]
}
```

### 4.4 获取周 K 线数据

- **接口路径**: `/stock/get_week_line`
- **请求方法**: GET
- **请求参数**:

```
code: string           // 股票代码
startDate: string      // 开始时间(可选)
endDate: string        // 结束时间(可选)
type: int              // 0不复权,1前复权,2后复权
```

- **响应示例**: 同日 K 线数据格式

### 4.5 获取月 K 线数据

- **接口路径**: `/stock/get_month_line`
- **请求方法**: GET
- **请求参数**:

```
code: string           // 股票代码
startDate: string      // 开始时间(可选)
endDate: string        // 结束时间(可选)
type: int              // 0不复权,1前复权,2后复权
```

- **响应示例**: 同日 K 线数据格式

### 4.6 获取分时数据

- **接口路径**: `/stock/get_stock_minute_data`
- **请求方法**: GET
- **请求参数**:

```
code: string  // 股票代码
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    ["09:30", "1790.00", "100"],
    ["09:31", "1792.00", "150"]
  ]
}
```

### 4.7 获取股票排行

- **接口路径**: `/stock/get_stock_rank`
- **请求方法**: GET
- **请求参数**:

```
node: string           // 股票市场/板块代码: {'a','b','ash','asz','bsh','bsz'} a(沪深A股)
industryCode: string   // 行业代码(可选)
pageIndex: int         // 页码
pageSize: int          // 每页大小
sort: string          // 排序字段: price,priceChange,pricePercent,buy,sell,open,close,high,low,volume,turnover
asc: int              // 排序方式: 0=降序(默认), 1=升序
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "code": "sh600519",
      "name": "贵州茅台",
      "price": "1800.00",
      "priceChange": "10.00",
      "changePercent": "0.56",
      "volume": "10000",
      "turnover": "18000000"
    }
  ]
}
```

### 4.8 获取指数代码列表

- **接口路径**: `/stock/get_index_code`
- **请求方法**: GET
- **请求参数**: 无

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "code": "sh000001",
      "name": "上证指数"
    }
  ]
}
```

### 4.9 获取行业代码列表

- **接口路径**: `/stock/get_industry_code`
- **请求方法**: GET
- **请求参数**: 无

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "code": "BK0001",
      "name": "银行"
    }
  ]
}
```

### 4.10 获取大盘数据

- **接口路径**: `/stock/get_board_info`
- **请求方法**: GET
- **请求参数**: 无

- **响应示例**:

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "code": "sh000001",
      "name": "上证指数",
      "price": "3000.00",
      "priceChange": "10.00",
      "changePercent": "0.33",
      "volume": "100000000",
      "turnover": "120000000000"
    }
  ]
}
```

## 5. 数据管理模块

### 5.1 下载数据

- **接口路径**: `/v1/data/download`
- **请求方法**: POST
- **请求参数**: 待实现

### 5.2 创建数据

- **接口路径**: `/v1/data/create`
- **请求方法**: POST
- **请求参数**: 待实现

### 5.3 上传数据

- **接口路径**: `/v1/data/upload`
- **请求方法**: POST
- **请求参数**: 待实现

### 5.4 删除数据

- **接口路径**: `/v1/data/delete`
- **请求方法**: POST
- **请求参数**: 待实现

## 6. 健康检查

- **接口路径**: `/v1/healthy`
- **请求方法**: GET
- **请求参数**: 无
- **响应**: 无具体内容，仅用于服务健康检查

## 7. 前端与后端交互示例

### 7.1 用户登录流程

1. 前端收集用户名和密码
2. 调用 `/v1/users/login` 接口
3. 后端验证用户名和密码
4. 返回登录结果，前端更新登录状态

### 7.2 智能对话流程

1. 前端获取或生成会话 ID
2. 用户输入问题，选择工具(可选)
3. 调用 `/v1/chat/` 接口，传递用户输入和会话 ID
4. 后端根据问题类型决定是否调用 MCP 工具
5. 流式返回对话结果，前端实时展示

### 7.3 股票数据查询流程

1. 用户在前端选择股票查询功能
2. 前端直接调用 `/stock/*` 相关 API
3. 后端调用外部股票 API 获取数据
4. 返回数据给前端进行可视化展示
