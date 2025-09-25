# Docker部署与Dify安装笔记

## 一、Docker与Docker Compose安装

### 1. 安装必要依赖

```bash
apt update
apt install -y ca-certificates curl gnupg lsb-release
```

### 2. 添加Docker GPG密钥

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | \
gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 3. 配置Docker软件源

```bash
echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
$(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 4. 安装Docker及Compose插件

```bash
apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 5. 验证安装

```bash
docker -v
docker compose version
```

## 二、配置Docker镜像加速

为解决Docker Hub拉取镜像慢或超时问题，配置国内镜像源：

### 1. 创建或编辑daemon.json配置文件

```bash
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
```

### 2. 添加以下内容到配置文件

```json
{
  "registry-mirrors": [
    "https://docker.211678.top",
    "https://docker.1panel.live",
    "https://hub.rat.dev",
    "https://docker.m.daocloud.io",
    "https://do.nark.eu.org",
    "https://dockerpull.com",
    "https://dockerproxy.cn",
    "https://docker.awsl9527.cn"
  ]
}
```

### 3. 重启Docker服务

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 4. 验证配置是否生效

```bash
docker info | grep Registry
```

## 三、部署Dify应用

### 1. 下载Dify源码

```bash
# 方法1：直接从GitHub克隆
git clone https://github.com/langgenius/dify.git
cd dify

# 方法2：下载ZIP包
wget https://github.com/langgenius/dify/archive/refs/heads/main.zip
unzip main.zip
cd dify-main
```

### 2. 创建环境变量文件

在docker目录下创建.env文件：

```bash
cd docker
nano .env
```

添加以下内容：

```
DB_USERNAME=dify
DB_PASSWORD=yourpassword
DB_DATABASE=dify
CERTBOT_EMAIL=your-email@example.com
CERTBOT_DOMAIN=your-domain.com
```

### 3. 启动Dify服务

```bash
docker compose up -d
```

如果遇到网络问题，确保已配置好镜像加速器，然后重试。

### 4. 初始化Dify管理员账号

在浏览器打开：
```
http://服务器IP/install
```

按照向导完成初始设置。

## 四、配置模型供应商

1. 登录Dify后台，点击右上角头像 → 插件 → Marketplace
2. 安装所需插件（如OpenAI-API-compatible）
3. 前往"设置 → 模型供应商"，配置API Key与地址

## 五、构建AI应用示例

可以创建空白应用，配置HTTP请求接入大模型API：

1. 创建应用，设置名称和描述
2. 添加HTTP请求节点，配置API地址和参数
3. 配置Headers和Body
4. 测试并发布应用

## 参考资料

- [从零开始：在Ubuntu上快速部署Docker和Dify](https://developer.aliyun.com/article/1680841)
- [Docker官方文档](https://docs.docker.com/)
- [Dify GitHub仓库](https://github.com/langgenius/dify)
