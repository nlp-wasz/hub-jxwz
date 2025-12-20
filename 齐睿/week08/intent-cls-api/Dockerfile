# 使用官方 Python 3.11 精简镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件，利用缓存层
COPY requirements.txt ./

# 安装系统依赖（编译器 + 网络证书）+ Python 依赖，然后清理缓存
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY . .

# 暴露端口（与 main.py 中 FastAPI 默认端口一致）
EXPOSE 8001
ENV DIFY_BASE_URL=http://172.17.0.1/v1
# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# 创建一个临时 Python 脚本
