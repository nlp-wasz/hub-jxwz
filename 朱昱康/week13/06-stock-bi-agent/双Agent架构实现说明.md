# 双 Agent 架构实现说明

## 概述

本文档说明了将股票分析 Agent 系统从单 agent 架构重构为双 agent 架构的实现细节。新架构包含两种专用 agent：闲聊 agent 和分析 agent，并实现了它们之间的无缝切换机制。

## 架构设计

### 1. 双 Agent 类型

#### 1.1 闲聊 Agent (Casual Agent)

- **功能**：专注于处理日常对话、情感交流及非结构化闲聊内容
- **特点**：
  - 保持对话自然流畅
  - 使用轻松愉快的语气
  - 避免专业术语
  - 提供情感支持和趣味性知识
- **名称**：小呆助手

#### 1.2 分析 Agent (Analysis Agent)

- **功能**：专注于处理数据分析、逻辑推理及结构化任务需求
- **特点**：
  - 使用专业严谨的术语
  - 基于数据和事实进行分析
  - 提供结构化结论
  - 支持工具调用和数据处理
- **名称**：小呆分析师

### 2. Agent 切换机制

#### 2.1 自动切换触发条件

**从闲聊 agent 切换到分析 agent**：
当用户输入包含以下关键词时触发切换：

- 数据分析类：分析、数据、股票、代码、图表、统计、趋势、预测等
- 金融投资类：计算、比较、评估、模型、算法、查询、报表、K 线等
- 专业术语类：市盈率、财报、投资、收益、风险、涨跌幅、成交量等

**从分析 agent 切换到闲聊 agent**：
当用户输入包含以下关键词时触发切换：

- 日常交流类：你好、谢谢、再见、天气、心情、感觉、喜欢、讨厌等
- 情感表达类：故事、笑话、聊天、闲聊、日常、生活、情感、兴趣等
- 休闲娱乐类：爱好、电影、音乐、美食、旅游、周末、假期、放松等

#### 2.2 切换流程

1. 用户输入消息
2. 系统检测当前 agent 类型
3. 根据关键词分析判断是否需要切换
4. 如需切换，记录切换事件并通知用户
5. 使用新 agent 处理用户输入
6. 保存对话记录和 agent 状态

### 3. 上下文保留机制

- 使用`AdvancedSQLiteSession`保存对话历史状态
- 切换 agent 时保留完整的对话上下文
- 系统消息中记录 agent 类型和切换事件
- 确保新 agent 能够理解之前的对话内容

## 实现细节

### 1. 文件结构

```
services/
  ├── chat.py              # 原始单agent实现
  └── chat_dual_agent.py    # 新双agent实现

routers/
  ├── chat.py              # 原始路由
  └── chat_dual_agent.py    # 新路由，支持双agent和手动切换
```

### 2. 关键函数

#### 2.1 Agent 类型检测

```python
def should_switch_to_analysis(content: str) -> bool:
    """判断是否应该从闲聊agent切换到分析agent"""
    analysis_keywords = [...]
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in analysis_keywords)

def should_switch_to_casual(content: str) -> bool:
    """判断是否应该从分析agent切换到闲聊agent"""
    casual_keywords = [...]
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in casual_keywords)
```

#### 2.2 Agent 类型决策

```python
def detect_agent_type(content: str, current_agent_type: str) -> str:
    """根据用户输入和当前agent类型，决定应该使用哪种agent"""
    if current_agent_type == "casual":
        return "analysis" if should_switch_to_analysis(content) else "casual"
    else:  # analysis
        return "casual" if should_switch_to_casual(content) else "analysis"
```

#### 2.3 对话处理流程

```python
async def chat(user_name:str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    # 获取当前agent类型
    current_agent_type = "casual"  # 默认使用闲聊agent

    # 检测是否需要切换agent
    new_agent_type = detect_agent_type(content, current_agent_type)

    # 如果agent类型发生变化，记录切换事件
    if new_agent_type != current_agent_type:
        switch_message = f"[系统] 已从{'闲聊' if current_agent_type == 'casual' else '分析'}模式切换到{'分析' if new_agent_type == 'analysis' else '闲聊'}模式"
        append_message2db(session_id, "system", switch_message)
        yield switch_message + "\n\n"
        current_agent_type = new_agent_type

    # 根据agent类型决定是否使用工具
    use_tools = (current_agent_type == "analysis" and tools and len(tools) > 0)

    # 使用相应的agent处理用户输入
    ...
```

### 3. 新增 API 接口

#### 3.1 手动切换 Agent

- **接口路径**: `/v1/chat/switch_agent`
- **请求方法**: POST
- **请求参数**:

```json
{
  "session_id": "string", // 会话ID
  "agent_type": "string" // "casual" 或 "analysis"
}
```

- **响应示例**:

```json
{
  "code": 200,
  "message": "Successfully switched to analysis agent",
  "data": {
    "agent_type": "analysis"
  }
}
```
