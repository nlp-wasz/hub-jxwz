# task01
id:cd3b2e51060366b0
<img width="336" height="122" alt="image" src="https://github.com/user-attachments/assets/2735d7db-fcfd-4215-a9ba-253783c2be14" />
谢谢老师！

# task02 Coze_playground
```python
"""
This example describes how to use the workflow interface to chat.
"""

import os
# Our official coze sdk for Python [cozepy](https://github.com/coze-dev/coze-py)
from cozepy import COZE_CN_BASE_URL

# Get an access_token through personal access token or oauth.
coze_api_token = 'cztei_q52SOOgcsJ2tVWa6Qvq8es3KepGXBN27Nh7r3KmTVLl1mBNAfPUocdnMQw5l84J5w'
# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
coze_api_base = COZE_CN_BASE_URL

from cozepy import Coze, TokenAuth, Message, ChatStatus, MessageContentType  # noqa

# Init the Coze client through the access_token.
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)

# Create a workflow instance in Coze, copy the last number from the web link as the workflow's ID.
workflow_id = '7561419554556526634'

# Call the coze.workflows.runs.create method to create a workflow run. The create method
# is a non-streaming chat and will return a WorkflowRunResult class.
workflow = coze.workflows.runs.create(
    workflow_id=workflow_id,
)

print("workflow.data", workflow.data)
```

# task03 本地fastapi部署Coze工作流
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cozepy import Coze, TokenAuth, COZE_CN_BASE_URL
import os
import json  

app = FastAPI(title="Coze Workflow Local Deploy")

COZE_API_TOKEN = os.getenv("COZE_API_TOKEN", "pat_gCvCrB5r8EOcJMRWqLyQX12TxODjRmJNN3pTuQia6Pi2tGXSFLwcRYgYFCkRrJZU")
COZE_API_BASE = COZE_CN_BASE_URL
DEFAULT_WORKFLOW_ID = "7561419554556526634"

# 初始化Coze客户端
coze = Coze(auth=TokenAuth(token=COZE_API_TOKEN), base_url=COZE_API_BASE)

# 请求模型定义
class WorkflowRequest(BaseModel):
    workflow_id: str = DEFAULT_WORKFLOW_ID
    query: str  # 用户查询内容，必填字段

# 响应模型定义
class WorkflowResponse(BaseModel):
    status: str
    data: dict = None  # 确保data为字典类型
    message: str = None
    workflow_id: str


@app.post("/run/workflow", response_model=WorkflowResponse)
async def run_workflow(request: WorkflowRequest):
    """运行Coze工作流，处理数据类型转换"""
    try:
        result = coze.workflows.runs.create(
            workflow_id=request.workflow_id,
            parameters={"query": request.query}
        )

        # 处理数据类型：如果是字符串则解析为JSON字典
        response_data = result.data
        if isinstance(response_data, str):
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError:
                response_data = {"raw_content": response_data}  # 保留原始内容

        return WorkflowResponse(
            status="success",
            data=response_data,  # 返回解析后的字典
            message="Workflow executed successfully",
            workflow_id=request.workflow_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

实现效果：![image](https://github.com/user-attachments/assets/cef3cbff-391e-4cc5-995e-2c639e3ca3a4)
