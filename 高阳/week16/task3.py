
from redisvl.extensions.router import SemanticRouter
from redisvl.extensions.router.schema import Route

# 1. 定义意图及其参考语句
routes = [
    Route(
        name="weather_query",
        references=[
            "今天天气怎么样?",
            "明天要下雨吗?",
            "外面多热啊?",
            "现在是天晴吗?",
            "告诉我本周的天气预报."
        ],
        distance_threshold=0.3  # 可选：为此意图设置特定的匹配阈值
    )
]

# 2. 初始化 SemanticRouter
router = SemanticRouter(
    name="intent_classifier",
    routes=routes,
    redis_url="redis://localhost:6379"
)


# 3. 执行意图识别
def identify_intent(user_input: str):
    route_match = router(user_input)
    if route_match.name:
        return route_match.name
    else:
        return "unknown_intent"


# 4. 使用示例
if __name__ == "__main__":
    test_inputs = [
        "北京的天气怎么样?",
        "明天天晴吗?",
        "天气预报",
        "未来一周要下雨吗？"  # 应该匹配不到或返回 unknown_intent
    ]

    for input_text in test_inputs:
        intent = identify_intent(input_text)
        print(f"Input: '{input_text}' -> Intent: {intent}")

