#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用MinerU云端API解析PDF/Word文档并实现RAG问答功能
"""

import os
import time
import requests
from typing import Dict, TypedDict
import uuid
import zipfile
import tempfile
from openai import OpenAI


class FileDict(TypedDict):
    name: str
    data_id: str


class MinerUDocumentParser:
    """基于MinerU的文档解析系统"""

    def __init__(self, api_key: str = None, api_url: str = "https://mineru.net"):
        """初始化文档解析系统

        Args:
            api_key: MinerU API密钥
            api_url: MinerU API地址
        """
        self.api_key = api_key or os.getenv("MINERU_API_KEY")
        self.api_url = api_url
        # 设置请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        } if self.api_key else {}

    def parse_document(self, file_path: list[str], files: list[FileDict]) -> Dict:
        """使用MinerU云端API解析文档
        Args:
            file_path: 本地文档路径
            files:文件属性
        Returns:
            解析结果字典
        """
        if not self.api_key:
            raise ValueError("请提供MinerU API密钥")

        data = {
            "files": files,
            "model_version": "pipeline"
        }

        try:
            response = requests.post(f"{self.api_url}/api/v4/file-urls/batch", headers=self.headers, json=data)
            if response.status_code == 200:
                result = response.json()
                print('response success. result:{}'.format(result))
                if result["code"] == 0:
                    batch_id = result["data"]["batch_id"]
                    urls = result["data"]["file_urls"]
                    print('batch_id:{},urls:{}'.format(batch_id, urls))
                    for i in range(0, len(urls)):
                        with open(file_path[i], 'rb') as f:
                            res_upload = requests.put(urls[i], data=f)
                            if res_upload.status_code == 200:
                                print(f"{urls[i]} upload success")
                            else:
                                print(f"{urls[i]} upload failed")
                    # 轮询任务状态
                    return self.poll_task_status(batch_id)
                else:
                    print('apply upload url failed,reason:{}'.format(result.msg))
            else:
                print('response not success. status:{} ,result:{}'.format(response.status_code, response))
        except Exception as err:
            print(err)

    def poll_task_status(self, batch_id: str, max_attempts: int = 30) -> Dict:
        """轮询任务状态直到完成
        
        Args:
            batch_id: 任务ID
            max_attempts: 最大尝试次数
            
        Returns:
            任务结果
        """
        url = f"{self.api_url}/api/v4/extract-results/batch/{batch_id}"

        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(max_attempts):
            try:
                result = requests.get(url, headers=header)
                print(result.json()["data"])
                if result.json()["code"] == 0 and result.json()["data"]["extract_result"][0]["state"] == "done":
                    return result.json()["data"]["extract_result"][0]["full_zip_url"]

            except requests.exceptions.RequestException as e:
                raise Exception(f"查询任务状态失败: {e}")

            # 等待一段时间再重试
            time.sleep(1)

        raise Exception("任务超时")


def extract_and_process_md(zip_content: bytes) -> str:
    """解压ZIP文件并提取full.md内容

    Args:
        zip_content: ZIP文件的二进制内容

    Returns:
        full.md文件的内容
    """
    # 创建临时目录处理ZIP文件
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "extracted.zip")

        # 保存ZIP内容到临时文件
        with open(zip_path, 'wb') as f:
            f.write(zip_content)

        # 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 查找full.md文件
        md_file_path = None
        for root, dirs, files in os.walk(temp_dir):
            if "full.md" in files:
                md_file_path = os.path.join(root, "full.md")
                break

        if md_file_path and os.path.exists(md_file_path):
            # 读取full.md内容
            with open(md_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise FileNotFoundError("未找到full.md文件")


def main():
    """主函数示例"""
    # 请替换为你的MinerU API密钥
    api_key = os.getenv("MINERU_API_KEY",
                        "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI4NzQwMDg2NyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2NTM2OTE5NCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiN2U5NTVjMmMtNDY4NC00MjI4LTg2YWItNTBkMWE2MDI2MWU2IiwiZW1haWwiOiIiLCJleHAiOjE3NjY1Nzg3OTR9.3ONYKKtbm8LfXQyKZ5FzPi20MJQ2019ZFRQHthy6Tdi94Tg7VgwX3M5GoFnNX-hFEVI35LDNk-TGjWyuZixxrg")

    # 创建文档解析实例
    parser = MinerUDocumentParser(api_key=api_key)

    # 示例：解析文档
    document_paths = [
        "/Users/msy/PycharmProjects/hub-jxwz/毛思扬/Week15/08-financial-report-generator/方案-队伍名字不能为空/方案-队伍名字不能为空.pdf"]
    files = []
    for document_path in document_paths:
        files.append({"name": os.path.basename(document_path), "data_id": str(uuid.uuid4()), "is_ocr": True})
    try:
        print("正在解析文档...")
        # 注意：这里我们只是演示代码结构，实际使用时需要真实的PDF文件
        # result = parser.parse_document(document_paths, files)
        result = parser.poll_task_status("62f0a852-8445-4544-83d3-d89f86865252")
        res_zip = requests.get(result)
        # 解压并获取full.md内容
        md_content = extract_and_process_md(res_zip.content)
        print("成功提取full.md内容")

        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key="sk-2ee5aadfe77245a4afcca79fc90c5931",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'system', 'content': '根据提供的信息文档回答问题，不要回答文档以外的问题。'},
                      {'role': 'system', 'content': f'以下是文档内容:{md_content}'},
                      {'role': 'user', 'content': '总结文档内容'}],
            stream=True,
            stream_options={"include_usage": True}
        )
        # 替换原有的打印JSON的代码
        for chunk in completion:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end='', flush=True)  # 实时打印内容而不换行


    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()
