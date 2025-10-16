# LLM文本信息抽取工具

本项目提供了两种基于大型语言模型（LLM）的文本信息抽取方法，用于从句子中识别领域、意图和实体。本项目对原始脚本进行了优化和改写，提升了代码质量、执行效率与功能稳定性，并使用FastAPI部署为Web服务。

## 项目结构

```
├── config/                    # 配置文件目录
│   ├── __init__.py
│   ├── settings.py            # 配置管理模块
│   └── .env                   # 环境变量配置
├── data/                      # 数据文件目录
│   ├── domains.txt            # 领域类别列表
│   ├── intents.txt            # 意图类别列表
│   ├── entities.txt           # 实体类别列表
│   └── sentences.txt          # 测试句子列表
├── src/                       # 源代码目录
│   ├── __init__.py
│   ├── extractors/           # 抽取器模块
│   │   ├── __init__.py
│   │   ├── prompt_engineering_extractor.py  # 提示词工程抽取器
│   │   └── agent_based_extractor.py         # 智能体抽取器
│   └── models/               # 数据模型（预留）
├── main.py                    # FastAPI应用入口
├── requirements.txt          # 项目依赖
└── README.md                  # 项目说明
```

## 功能概述

本工具可以从输入的文本中抽取以下信息：
1. **领域类别**：识别文本所属的领域，如音乐、应用、天气等
2. **意图类别**：识别用户的意图，如打开、搜索、查询等
3. **实体标签**：识别并分类文本中的实体，如地点、人名、时间等

## 两种抽取方法

### 1. 提示词工程方法

- **位置**: `src/extractors/prompt_engineering_extractor.py`
- **原理**: 通过精心设计的提示词，引导模型直接输出结构化的JSON结果
- **特点**: 实现简单，直观易懂

### 2. 智能体方法

- **位置**: `src/extractors/agent_based_extractor.py`
- **原理**: 利用函数调用（Function Calling）机制，让模型通过调用预定义的函数来输出结构化结果
- **特点**: 结构化输出更可靠，适合复杂场景

## API使用

### 启动服务

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

服务将在 `http://localhost:8000` 启动，可以通过 `http://localhost:8000/docs` 访问API文档。

### API接口

#### 1. 文本信息抽取接口

- **路径**: `/extract`
- **方法**: POST
- **请求体**:
```json
{
  "text": "查询许昌到中山的汽车。",
  "method": "prompt_engineering"  // 或 "agent_based"
}
```

- **响应**:
```json
{
  "success": true,
  "message": "信息抽取成功",
  "data": {
    "domain": "bus",
    "intent": "QUERY",
    "slots": {
      "startLoc_city": "许昌",
      "endLoc_city": "中山"
    }
  },
  "method": "prompt_engineering"
}
```

#### 2. 健康检查接口

- **路径**: `/health`
- **方法**: GET
- **响应**:
```json
{
  "status": "healthy",
  "message": "服务运行正常"
}
```

### 使用示例

#### Python请求示例

```python
import requests

# API地址
url = "http://localhost:8000/extract"

# 请求体
data = {
    "text": "查询许昌到中山的汽车。",
    "method": "prompt_engineering"
}

# 发送请求
response = requests.post(url, json=data)

# 解析响应
if response.status_code == 200:
    result = response.json()
    if result["success"]:
        print("抽取结果:", result["data"])
    else:
        print("抽取失败:", result["message"])
else:
    print("请求失败:", response.status_code)
```

#### curl请求示例

```bash
# 使用提示词工程方法
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{"text": "查询许昌到中山的汽车。", "method": "prompt_engineering"}'

# 使用智能体方法
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{"text": "查询许昌到中山的汽车。", "method": "agent_based"}'
```

## 配置说明

### 环境变量配置

在 `config/.env` 文件中可以配置以下参数：

```env
# OpenAI API配置
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 模型配置
DEFAULT_MODEL=qwen-plus

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 日志配置
LOG_LEVEL=INFO
```

### 数据文件说明

- `domains.txt`: 领域类别列表，每行一个类别
- `intents.txt`: 意图类别列表，每行一个类别
- `entities.txt`: 实体类别列表，每行一个类别
- `sentences.txt`: 测试句子列表，每行一个句子
