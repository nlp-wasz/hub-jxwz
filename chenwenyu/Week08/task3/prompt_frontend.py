# prompt_frontend.py
import requests
import json

class SentenceAnalyzerClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def check_service_status(self):
        """检查服务状态和客户端初始化状态"""
        try:
            health_response = requests.get(f"{self.base_url}/health")
            health_data = health_response.json()
            
            return {
                "service_running": True,
                "client_initialized": health_data.get("client_initialized", False),
                "status": health_data.get("status"),
                "message": health_data.get("message", "")
            }
        except requests.exceptions.RequestException:
            return {
                "service_running": False,
                "client_initialized": False,
                "message": "后端服务未启动"
            }
    
    def analyze_sentence(self, sentence: str):
        """分析单个句子"""
        url = f"{self.base_url}/analyze"
        data = {"sentence": sentence}
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"请求失败: {e}"}

# 简单的命令行前端
def main():
    client = SentenceAnalyzerClient()
    
    print("🔍 句子分析工具")
    print("=" * 40)
    
    # 检查服务状态
    status = client.check_service_status()
    
    if not status["service_running"]:
        print("❌ 后端服务未启动，请先运行: python backend.py")
        return
    
    if not status["client_initialized"]:
        print("❌ ZhipuAI客户端未初始化")
        print("💡 请设置环境变量: export ZHIPUAI_API_KEY=your_api_key")
        return
    
    print("✅ 服务状态正常，可以开始分析句子")
    print("💡 输入 'quit' 或 'exit' 退出程序")
    print("-" * 40)
    
    while True:
        sentence = input("\n请输入要分析的句子: ").strip()
        
        if sentence.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
            
        if not sentence:
            print("❌ 句子不能为空，请重新输入")
            continue
        
        print("分析中...")
        result = client.analyze_sentence(sentence)
        
        if "error" in result:
            print(f"❌ 分析失败: {result['error']}")
        else:
            print("\n📊 分析结果:")
            print(f"   句子: {result['sentence']}")
            print(f"   领域: {result['domain']}")
            print(f"   意图: {result['intent']}")
            if result['slots']:
                print(f"   实体: {json.dumps(result['slots'], ensure_ascii=False, indent=4)}")
            else:
                print("   实体: 无")

if __name__ == "__main__":
    main()