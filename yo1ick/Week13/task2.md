# 接口文档

## 一、接口统计
共31个前后端交互接口，按模块划分如下：
- **chat模块**：6个
- **stock模块**：4个  
- **user模块**：7个  
- **data模块**：4个  
- **autostock模块**：10个  
- **news/saying/tool模块**：MCP服务接口（非HTTP接口）

## 二、核心接口详情

### 1. 聊天模块（/v1/chat）
| 接口路径 | 方法 | 描述 | 请求参数 |
|----------|------|------|----------|
| `/` | POST | 流式聊天接口 | RequestForChat（user_name, task, session_id, content, tools） |
| `/init` | POST | 初始化聊天会话 | - |
| `/get` | POST | 获取单一会话记录 | session_id |
| `/delete` | POST | 删除会话 | session_id |
| `/list` | POST | 获取用户会话列表 | user_name |
| `/feedback` | POST | 消息反馈 | session_id, message_id, feedback |

### 2. 用户模块（/v1/users）
| 接口路径 | 方法 | 描述 | 请求参数 |
|----------|------|------|----------|
| `/login` | POST | 用户登录 | RequestForUserLogin（user_name, password） |
| `/register` | POST | 用户注册 | RequestForUserRegister（user_name, password, user_role） |
| `/reset-password` | POST | 密码重置 | RequestForUserResetPassword（user_name, password, new_password） |
| `/info` | POST | 获取用户信息 | user_name |
| `/reset-info` | POST | 修改用户信息 | RequestForUserChangeInfo（user_name, user_role, status） |
| `/delete` | POST | 删除用户 | user_name |
| `/list` | POST | 获取用户列表 | - |

### 3. 股票模块（/v1/stock）
| 接口路径 | 方法 | 描述 | 请求参数 |
|----------|------|------|----------|
| `/list_fav_stock` | POST | 获取用户收藏股票 | user_name |
| `/del_fav_stock` | POST | 删除收藏股票 | user_name, stock_code |
| `/add_fav_stock` | POST | 添加收藏股票 | user_name, stock_code |
| `/clear_fav_stock` | POST | 清空收藏股票 | user_name |

### 4. 数据模块（/v1/data）
| 接口路径 | 方法 | 描述 | 请求参数 |
|----------|------|------|----------|
| `/download` | POST | 下载数据 | - |
| `/create` | POST | 创建数据 | - |
| `/upload` | POST | 上传数据 | - |
| `/delete` | POST | 删除数据 | - |

### 5. 底层股票接口（/stock）
| 接口路径 | 方法 | 描述 | 请求参数 |
|----------|------|------|----------|
| `/get_stock_code` | GET | 股票代码查询 | keyword（可选） |
| `/get_index_code` | GET | 指数代码查询 | - |
| `/get_industry_code` | GET | 行业代码查询 | - |
| `/get_board_info` | GET | 大盘数据查询 | - |
| `/get_stock_rank` | GET | 股票排行 | node, industryCode（可选）, pageIndex, pageSize, sort, asc |
| `/get_month_line` | GET | 月K线数据 | code, startDate（可选）, endDate（可选）, type |
| `/get_week_line` | GET | 周K线数据 | code, startDate（可选）, endDate（可选）, type |
| `/get_day_line` | GET | 日K线数据 | code, startDate（可选）, endDate（可选）, type |
| `/get_stock_info` | GET | 股票基础信息 | code |
| `/get_stock_minute_data` | GET | 分时数据 | code |

## 三、MCP服务接口
news/saying/tool模块通过FastMCP提供工具调用能力，包含以下核心工具：
- **news.py**：get_today_daily_news, get_douyin_hot_news等5个新闻工具
- **saying.py**：get_today_familous_saying等3个名言工具  
- **tool.py**：get_city_weather等6个实用工具
