# 接口文档

## 1. 聊天接口 (Chat API)

基础路径: /v1/chat

### 1.1 流式聊天

• 端点: POST /

• 功能: 进行流式聊天对话

• 请求体: RequestForChat
  {
    "user_name": "string",
    "task": "string",
    "session_id": "string",
    "content": "string",
    "tools": "string"
  }
  
• 响应: StreamingResponse (text/event-stream)

### 1.2 初始化聊天会话

• 端点: POST /init

• 功能: 生成新的聊天会话ID

• 响应: 
  {
    "code": 200,
    "message": "ok",
    "data": {
      "session_id": "随机生成的会话ID"
    }
  }
  

### 1.3 获取聊天记录

• 端点: POST /get

• 参数: session_id: str

• 功能: 根据会话ID获取聊天记录

### 1.4 删除聊天记录

• 端点: POST /delete

• 参数: session_id: str

• 功能: 删除指定会话的聊天记录

### 1.5 列出用户聊天

• 端点: POST /list

• 参数: user_name: str

• 功能: 列出用户的所有聊天记录

### 1.6 消息反馈

• 端点: POST /feedback

• 参数: session_id: str, message_id: int, feedback: bool

• 功能: 对特定消息进行点赞/点踩反馈

## 2. 数据管理接口 (Data API)

基础路径: /v1/data

### 2.1 下载数据

• 端点: POST /download

• 功能: 下载数据文件

### 2.2 创建数据

• 端点: POST /create

• 功能: 创建新的数据

### 2.3 上传数据

• 端点: POST /upload

• 功能: 上传数据文件

### 2.4 删除数据

• 端点: POST /delete

• 功能: 删除数据

## 3. 股票接口 (Stock API)

基础路径: /v1/stock

### 3.1 获取用户收藏股票

• 端点: POST /list_fav_stock

• 参数: user_name: str

• 功能: 获取用户收藏的所有股票

### 3.2 删除收藏股票

• 端点: POST /del_fav_stock

• 参数: user_name: str, stock_code: str

• 功能: 删除用户收藏的指定股票

### 3.3 添加收藏股票

• 端点: POST /add_fav_stock

• 参数: user_name: str, stock_code: str

• 功能: 添加股票到用户收藏

### 3.4 清空收藏股票

• 端点: POST /clear_fav_stock

• 参数: user_name: str

• 功能: 清空用户的所有收藏股票

## 4. 用户管理接口 (Users API)

基础路径: /v1/users

### 4.1 用户登录

• 端点: POST /login

• 请求体: RequestForUserLogin
  {
    "user_name": "string",
    "password": "string"
  }
  

### 4.2 用户注册

• 端点: POST /register

• 请求体: RequestForUserRegister
  {
    "user_name": "string",
    "password": "string",
    "user_role": "string"
  }
  

### 4.3 重置密码

• 端点: POST /reset-password

• 请求体: RequestForUserResetPassword
  {
    "user_name": "string",
    "password": "string",
    "new_password": "string"
  }
  

### 4.4 获取用户信息

• 端点: POST /info

• 参数: user_name: str

• 功能: 获取指定用户的基本信息

### 4.5 修改用户信息

• 端点: POST /reset-info

• 请求体: RequestForUserChangeInfo
  {
    "user_name": "string",
    "user_role": "string",  // 可选
    "status": "string"      // 可选
  }
  

### 4.6 删除用户

• 端点: POST /delete

• 参数: user_name: str

• 功能: 删除指定用户

### 4.7 用户列表

• 端点: POST /list

• 功能: 获取所有用户列表
