"""
启动所有分类的MCP服务器
"""
import asyncio
import subprocess
import sys
import time
import signal
import os

# 服务器配置
SERVERS = [
    {"name": "主服务器", "port": 8900, "script": "mcp_server_main.py"},
    {"name": "新闻服务器", "port": 8901, "script": "news_server.py"},
    {"name": "工具服务器", "port": 8902, "script": "tools_server.py"},
    {"name": "名言服务器", "port": 8903, "script": "saying_server.py"},
    {"name": "情感分析服务器", "port": 8904, "script": "sentiment_server.py"}
]

# 保存子进程引用
processes = []

def signal_handler(sig, frame):
    """处理中断信号，关闭所有子进程"""
    print("\n正在关闭所有服务器...")
    for process in processes:
        if process.poll() is None:  # 进程仍在运行
            process.terminate()
    sys.exit(0)

def create_server_script(server_type, port):
    """创建特定类型的服务器脚本"""
    script_content = f"""
import asyncio
from fastmcp import FastMCP
# 导入对应的MCP模块
from {server_type} import mcp as {server_type}_mcp
# 创建服务器
{server_type}_mcp_server = FastMCP(name="{server_type.title()}-MCP-Server")
async def setup():
    await {server_type}_mcp_server.import_server({server_type}_mcp, prefix="")
if __name__ == "__main__":
    asyncio.run(setup())
    {server_type}_mcp_server.run(transport="sse", port={port})
"""

    script_name = f"{server_type}_server.py"
    with open(script_name, "w") as f:
        f.write(script_content)

    return script_name

def main():
    """主函数"""
    print("启动企业职能助手MCP服务器...")

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 创建并启动服务器脚本
    for i, server in enumerate(SERVERS):
        if server["name"] == "主服务器":
            # 主服务器已经存在，直接启动
            cmd = [sys.executable, server["script"]]
        else:
            # 创建其他服务器脚本
            server_type = server["name"].replace("服务器", "")
            script_name = create_server_script(server_type.lower(), server["port"])
            cmd = [sys.executable, script_name]

        print(f"启动 {server['name']} (端口: {server['port']})...")

        # 启动子进程
        process = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        processes.append(process)

        # 等待一小段时间，确保服务器启动
        time.sleep(1)

    print("\n所有服务器已启动！")
    print("按 Ctrl+C 关闭所有服务器")

    # 等待所有进程
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()