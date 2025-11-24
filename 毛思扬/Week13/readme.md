# 作业1:
复现一下chatbi项目，三部分（mcp、fastapi、streamlit）组成，本地启动运行。
![截屏2025-11-24 18.24.26.png](%E6%88%AA%E5%B1%8F2025-11-24%2018.24.26.png)

# 作业2:
chatbi的项目梳理一下，现在前端和后端交互接口有多少个，写一个接口文档出来，提交。
根据检索到的信息，我已经梳理了chatbi项目的前后端交互接口，并整理成以下接口文档：

## ChatBI项目接口文档

### 1. 用户管理接口

#### 用户登录
- **URL**: `/v1/users/login`
- **Method**: POST
- **Request Body**: 
  ```json
  {
    "user_name": "string",
    "password": "string"
  }
  ```

- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

#### 用户注册
- **URL**: `/v1/users/register`
- **Method**: POST
- **Request Body**: 
  ```json
  {
    "user_name": "string",
    "password": "string",
    "user_role": "string"
  }
  ```

- **Response**: `BasicResponse`

#### 用户信息获取
- **URL**: `/v1/users/info`
- **Method**: POST
- **Parameters**: `user_name` (query)
- **Response**: `BasicResponse`

#### 用户密码重置
- **URL**: `/v1/users/reset_password`
- **Method**: POST
- **Request Body**: 
  ```json
  {
    "user_name": "string",
    "password": "string",
    "new_password": "string"
  }
  ```

- **Response**: `BasicResponse`

#### 用户信息修改
- **URL**: `/v1/users/change_info`
- **Method**: POST
- **Request Body**: 
  ```json
  {
    "user_name": "string",
    "user_role": "string (optional)",
    "status": "boolean (optional)"
  }
  ```

- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

### 2. 聊天接口

#### 初始化聊天
- **URL**: `/v1/chat/init`
- **Method**: POST
- **Response**: `StreamingResponse`

#### 发送聊天消息
- **URL**: `/v1/chat/send`
- **Method**: POST
- **Request Body**: 
  ```json
  {
    "content": "string",
    "user_name": "string",
    "session_id": "string (optional)",
    "task": "string (optional)",
    "tools": ["string"] (optional)
  }
  ```

- **Response**: `StreamingResponse`

#### 获取聊天记录
- **URL**: `/v1/chat/get`
- **Method**: POST
- **Parameters**: [session_id](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L40-L40) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

#### 获取聊天列表
- **URL**: `/v1/chat/list`
- **Method**: POST
- **Parameters**: [user_name](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L12-L12) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

#### 删除聊天记录
- **URL**: `/v1/chat/delete`
- **Method**: POST
- **Parameters**: [session_id](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L40-L40) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

#### 反馈聊天记录
- **URL**: `/v1/chat/feedback`
- **Method**: POST
- **Parameters**: [session_id](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L40-L40), [feedback](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L57-L57) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

### 3. 股票数据接口

#### 获取用户收藏股票列表
- **URL**: `/v1/stock/list_fav_stock`
- **Method**: POST
- **Parameters**: [user_name](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L12-L12) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

#### 添加用户收藏股票
- **URL**: `/v1/stock/add_fav_stock`
- **Method**: POST
- **Parameters**: [user_name](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L12-L12), [stock_code](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L63-L63) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

#### 删除用户收藏股票
- **URL**: `/v1/stock/del_fav_stock`
- **Method**: POST
- **Parameters**: [user_name](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L12-L12), [stock_code](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L63-L63) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

#### 清空用户收藏股票
- **URL**: `/v1/stock/clear_fav_stock`
- **Method**: POST
- **Parameters**: [user_name](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/orm.py#L12-L12) (query)
- **Response**: [BasicResponse](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L11-L14)

### 4. 底层股票API接口

这些接口通过 `/stock` 路径挂载：

#### 获取股票排名
- **URL**: `/stock/get_stock_rank`
- **Method**: GET
- **Parameters**: `node`, `pageIndex`, `pageSize`, `sort`, `asc`, `industryCode` (optional)

#### 获取股票信息
- **URL**: `/stock/get_stock_info`
- **Method**: GET
- **Parameters**: [code](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L12-L12)

#### 获取股票分钟数据
- **URL**: `/stock/get_stock_minute_data`
- **Method**: GET
- **Parameters**: [code](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L12-L12)

#### 获取股票日K线
- **URL**: `/stock/get_stock_day_kline`
- **Method**: GET
- **Parameters**: [code](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L12-L12), `startDate`, `endDate`, [type](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week12/3-agent复杂案例/35_Tool-ComputerUse.py#L140-L141)

#### 获取股票周K线
- **URL**: `/stock/get_stock_week_kline`
- **Method**: GET
- **Parameters**: [code](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week13/06-stock-bi-agent/models/data_models.py#L12-L12), `startDate`, `endDate`, [type](file:///Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week12/3-agent复杂案例/35_Tool-ComputerUse.py#L140-L141)

#### 获取行业代码
- **URL**: `/stock/get_industry_code`
- **Method**: GET

### 总结

 **12个主要业务接口** 和 **7个底层股票API接口**

# 作业3:
在service中，定义两种agent，一种是闲聊agent，一种股票agent，通过handoff链接一起进行对话。

![截屏2025-11-24 21.32.18.png](%E6%88%AA%E5%B1%8F2025-11-24%2021.32.18.png)
