import uvicorn
import json
from fastapi import FastAPI
from question_analyze_prompt import question_analyze_prompt
from question_analyze_tools import QuestionAnalyzeAgent

app = FastAPI(title="Question Analyze")

@app.get("/prompt")
async def prompt(question: str):
    return json.loads(question_analyze_prompt(question)[0].message.content)

@app.get("/tools")
async def tools(question: str):
    return QuestionAnalyzeAgent().call(question)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
    # http://127.0.0.1:8000/docs