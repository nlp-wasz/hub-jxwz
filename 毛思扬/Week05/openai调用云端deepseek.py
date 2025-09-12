# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

# 初始化客户端
client = OpenAI(api_key="sk-4eb3d6029c3a45e7a166cb1dd3331e18", base_url="https://api.deepseek.com")

# 系统提示词
system_prompt = """你现在是一位专业的旅行规划师，你的责任是根据旅行目的地、
天数、行程风格（紧凑、适中、休闲），帮助我规划旅游行程并生成旅行计划表。
请你以表格的方式呈现结果。 旅行计划表的表头请包含日期、地点、行程计划、交通方式、备注。所有表头都为必填项，
请加深思考过程，严格遵守以下规则： 1. 日期请以DayN为格式如Day1。 2. 地点需要呈现当天所在城市，请根据日期、
考虑地点的地理位置远近，严格且合理制定地点。 3. 行程计划需包含位置、时间、活动，其中位置需要根据地理位置的远近进行排序，
位置的数量可以根据行程风格灵活调整，如休闲则位置数量较少、紧凑则位置数量较多，时间需要按照上午、中午、
晚上制定并给出每一个位置所停留的时间如上午10点-中午12点，活动需要准确描述在位置发生的对应活动如参观xxx、游玩xxx、
吃饭等，需根据位置停留时间合理安排活动类型。 4. 交通方式需根据地点、行程计划中的每个位置的地理距离合理选择步行、地铁、
飞机等不同的交通方式。 5. 备注中需要包括对应行程计划需要考虑到的注意事项。 现在请你严格遵守以上规则，根据我的旅行目的
地、天数、行程风格（紧凑、适中、休闲），再以表格的方式生成合理的旅行计划表，提供表格后请再询问我行程风格、偏好、特殊要
求等，并根据此信息完善和优化旅行计划表再次提供，直到我满意。记住你要根据我提供的旅行目的地、天数等信息以表格形式生成旅
行计划表，最终答案一定是表格形式"""


class ChatBot:
    def __init__(self):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.is_responding = False

    def stream_response(self, user_input):
        """流式获取模型响应"""
        # 将用户消息添加到对话历史
        self.messages.append({"role": "user", "content": user_input})

        # 初始化响应内容
        response_content = ""

        try:
            # 使用流式调用
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                stream=True
            )

            # 逐块接收并打印响应
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    response_content += content

        except Exception as e:
            print(f"\n错误: {e}")
            return ""

        # 换行
        print()
        return response_content

    def handle_user_input(self):
        """处理用户输入"""
        while True:
            # 显示输入提示
            if not self.is_responding:
                user_input = input("\n您: ")

                # 检查退出条件
                if user_input.lower() in ['退出', 'quit', 'exit']:
                    print("助手: 感谢使用旅行规划助手，祝您旅途愉快！")
                    break

                # 设置正在响应标志
                self.is_responding = True
                print("助手: ", end="", flush=True)

                # 获取模型响应
                response_content = self.stream_response(user_input)

                # 将助手的完整回复添加到对话历史
                if response_content:
                    self.messages.append({"role": "assistant", "content": response_content})

                # 重置响应标志
                self.is_responding = False
            else:
                # 如果正在响应，则短暂等待
                import time
                time.sleep(0.1)


def chat_with_model():
    print("旅行规划助手已启动！请输入您的旅行需求，输入 '退出' 或 'quit' 结束对话。")
    print("-" * 50)
    print("提示：模型正在生成响应时，您的输入将被忽略。")

    bot = ChatBot()
    bot.handle_user_input()


if __name__ == "__main__":
    chat_with_model()
