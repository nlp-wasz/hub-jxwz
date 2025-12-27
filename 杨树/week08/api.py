from fastapi import FastAPI

from prompt import analyze_user_input_prompt
from tools import analyze_user_input_tools

app = FastAPI(
    title="识别",
)

@app.get("/prompt")
async def prompt(inputString: str):
    return analyze_user_input_prompt(inputString)


@app.get("/tools")
async def prompt(inputString: str):
    return analyze_user_input_tools(inputString)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
