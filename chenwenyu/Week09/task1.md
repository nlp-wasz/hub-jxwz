
# 安装Docker，Dify，配置Dify与本地Elasticsearch连接

# 查看docker中的dify服务
```bash
cd ~/work/dify/docker
docker compose ps

NAME                     IMAGE                                       COMMAND                  SERVICE         CREATED        STATUS                  PORTS
docker-api-1             langgenius/dify-api:1.9.1                   "/bin/bash -c 'uv pi…"   api             26 hours ago   Up 26 hours             5001/tcp
docker-db-1              postgres:15-alpine                          "docker-entrypoint.s…"   db              26 hours ago   Up 26 hours (healthy)   5432/tcp
docker-nginx-1           nginx:latest                                "sh -c 'cp /docker-e…"   nginx           26 hours ago   Up 26 hours             0.0.0.0:80->80/tcp, [::]:80->80/tcp, 0.0.0.0:443->443/tcp, [::]:443->443/tcp
docker-plugin_daemon-1   langgenius/dify-plugin-daemon:0.3.0-local   "/bin/bash -c /app/e…"   plugin_daemon   26 hours ago   Up 26 hours             0.0.0.0:5003->5003/tcp, [::]:5003->5003/tcp
docker-redis-1           redis:6-alpine                              "docker-entrypoint.s…"   redis           26 hours ago   Up 26 hours (healthy)   6379/tcp
docker-sandbox-1         langgenius/dify-sandbox:0.2.12              "/main"                  sandbox         26 hours ago   Up 26 hours (healthy)
docker-ssrf_proxy-1      ubuntu/squid:latest                         "sh -c 'cp /docker-e…"   ssrf_proxy      26 hours ago   Up 26 hours             3128/tcp
docker-web-1             langgenius/dify-web:1.9.1                   "/bin/sh ./entrypoin…"   web             26 hours ago   Up 26 hours             3000/tcp
docker-worker-1          langgenius/dify-api:1.9.1                   "/bin/bash -c 'uv pi…"   worker          26 hours ago   Up 26 hours             5001/tcp
docker-worker_beat-1     langgenius/dify-api:1.9.1                   "/bin/bash /entrypoi…"   worker_beat     26 hours ago   Up 26 hours             5001/tcp
(langchain_venv) wenyuc 10-22 15:17 ~/work/dify/docker>
```

# 测试dify与本地Elasticsearch向量数据库的连接
```bash
cd ~/work/dify/docker
docker compose exec api curl -u <user>:<password> "http://host.docker.internal:9200/_cat/indices?v&h=index,health,status,docs.count,store.size,creation.date.string"|grep -v '^\..*' | column -t
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  2940    0  2940    0     0  35087      0 --:--:-- --:--:-- --:--:-- 35421
index                                                   health  status  docs.count  store.size  creation.date.string
semantic_search_demo                                    yellow  open    5           68.8kb      2025-09-18T03:40:42.752Z
vector_index_88436271_8a71_426c_828e_45fc434a4e3d_node  yellow  open    85          1.9mb       2025-10-22T03:50:03.057Z
vector_index_5050a2b4_9910_4f27_832c_2612019c0db6_node  yellow  open    66          1.4mb       2025-10-21T05:40:55.415Z
vector_index_5781acd9_1222_446f_8a41_7e6609201c7a_node  yellow  open    10          323.3kb     2025-10-22T06:49:26.265Z
kibana_sample_data_flights                              yellow  open    13014       5.8mb       2025-09-15T07:48:51.636Z
deeplearning-theses                                     yellow  open    138         1.2mb       2025-09-19T03:13:04.717Z
products                                                yellow  open    2           7.2kb       2025-09-18T03:40:04.919Z
```
以vector_index_开头的index都是在dify中创建知识库生成的index

# workflow 和chatflow的区别
chatflow是具有记忆能力的多轮对话任务的工作流程
它的特点有：支持多轮复杂对话，具备对话记忆功能，适用于需要上下文维护的对话场景

workflow用于单轮或多轮的工作流程，如自动化和批处理
它的特点有：专注于单轮或多轮任务，支持自动化流程；适用于批处理的任务。
