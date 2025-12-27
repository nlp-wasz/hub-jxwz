#!/usr/bin/env python3
# animal_recognizer.py
import os
import base64
import mimetypes
import argparse
import dotenv
import dashscope
from dashscope import MultiModalConversation

dotenv.load_dotenv()

def image_to_base64(path: str) -> str:
    """本地图片→data-url"""
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

def build_message(image_source: str) -> dict:
    """构造多轮消息"""
    # 如果是本地文件
    if os.path.isfile(image_source):
        image_url = image_to_base64(image_source)
    else:  # 直接当远程地址用
        image_url = image_source

    return {
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": "请用中文回答：1）图中有什么小动物？2）用一句话简单介绍它（们）。"}
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 小动物识别")
    # parser.add_argument("--image", required=True, help="本地路径或 http(s) 图片地址")
    parser.add_argument("--image", default="./pic/cat.png", help="本地路径或 http(s) 图片地址")
    parser.add_argument("--model", default="qwen3-vl-plus", help="模型名，默认 qwen3-vl-plus")
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY 或写入 .env 文件")

    messages = [build_message(args.image)]

    # 流式调用
    response = MultiModalConversation.call(
        api_key=api_key,
        model=args.model,
        messages=messages,
        stream=True,
        enable_thinking=False  # 只做识别，不需要思考过程
    )

    print("=== 识别结果 ===")
    for chunk in response:
        text = chunk.output.choices[0].message.get("content", [])
        if text:
            print(text[0]["text"], end="")
    print()

if __name__ == "__main__":
    main()
    # 运行：python qwen-vl_animal_recognizer.py --image ./pic/more_animal_cartoon.png
    # 小猫：
    # 1）图中是一只猫，看起来像是孟加拉猫或带有豹纹斑点的混种猫。
    # 2）它有着金棕色带深色条纹的皮毛和碧绿色的眼睛，正慵懒地趴在沙发上，表情高冷又可爱。
    # 小狗：
    # 1）图中是一只狗。
    # 2）这是一只毛色棕黄、耳朵竖立的土狗，正抬头望向远方，神情温顺而警觉。
    # 卡通多动物：
    # 1）图中是多种可爱卡通小动物，包括：兔子、猪、猫头鹰、熊猫、老虎、兔子、猪、熊、熊猫、狮子、长颈鹿、猫鳄鱼、猪、狗、熊、考拉、熊、企鹅、斑马。
    # 2）这些小动物都以萌趣可爱的卡通形象呈现，色彩鲜艳、表情生动，适合用于儿童设计或装饰素材。
    # 真实多动物：（但只识别出了一种,改前提示词：请用中文回答：1）图中是什么小动物？2）用一句话简单介绍它。）
    # 1）图中右上角的小动物是浣熊。
    # 2）浣熊是一种夜行性哺乳动物，以其“戴面具”的脸和灵巧的前爪著称，常在水边觅食，聪明又好奇。
    # 改后提示词：请用中文回答：1）图中有什么小动物？2）用一句话简单介绍它（们）。
    # 1）图中有哪些小动物？
    # 图中共有四种动物：
    # - 长颈鹿（左上角）
    # - 浣熊（右上角）
    # - 棕熊（左下角）
    # - 火烈鸟和白鹭（右下角）
    #
    # 2）用一句话简单介绍它们：
    # 长颈鹿优雅漫步于林间，以高挑身姿俯瞰大地；浣熊手捧食物，机灵可爱地享受美食；棕熊在水中嬉戏，憨态可掬地玩着红色浮圈；火烈鸟与白鹭在水边悠然伫立，粉白相映成趣，构成一幅宁静的湿地画
    # 卷。


    # 小猫链接：https://image.baidu.com/search/detail?adpicid=0&b_applid=9370891442632145004&bdtype=0&commodity=&copyright=&cs=3070006451%2C3342865523&di=7552572858984038401&fr=click-pic&fromurl=http%253A%252F%252Fbaijiahao.baidu.com%252Fs%253Fid%253D1823644970417353653%2526wfr%253Dspider%2526for%253Dpc&gsm=1e&hd=&height=0&hot=&ic=&ie=utf-8&imgformat=&imgratio=&imgspn=0&is=3688259323%2C3873907483&isImgSet=&latest=&lid=&lm=&objurl=https%253A%252F%252Fpic.rmb.bdstatic.com%252Fbjh%252F250210%252Fdump%252F398f5e142c67867ad5de13f53fbc90a6.jpeg&os=3688259323%2C3873907483&pd=image_content&pi=0&pn=13&rn=1&simid=47550718%2C653903118&tn=baiduimagedetail&width=0&word=%E5%B0%8F%E7%8C%AB&z=
    # 各种小动物链接：
    # https://image.baidu.com/search/detail?adpicid=0&b_applid=8897018148669183546&bdtype=0&commodity=&copyright=&cs=2606840430%2C951668386&di=7552572858984038401&fr=click-pic&fromurl=http%253A%252F%252Fwww.douyin.com%252Fnote%252F7360610919962987828&gsm=3c&hd=&height=0&hot=&ic=&ie=utf-8&imgformat=&imgratio=&imgspn=0&is=2165869281%2C933145697&isImgSet=&latest=&lid=&lm=&objurl=https%253A%252F%252Fp3-pc-sign.douyinpic.com%252Ftos-cn-i-0813c001%252FoYDJKAAImAA27ftmCVeDgSnhAJA9YGAIESbhNi~tplv-dy-aweme-images%253Aq75.webp&os=2165869281%2C933145697&pd=image_content&pi=0&pn=30&rn=1&simid=2606840430%2C951668386&tn=baiduimagedetail&width=0&word=%E5%90%84%E7%A7%8D%E5%B0%8F%E5%8A%A8%E7%89%A9%E5%9B%BE%E7%89%87&z=
    # https://image.baidu.com/search/detail?adpicid=0&b_applid=11245500017902750374&bdtype=0&commodity=&copyright=&cs=2151437926%2C3227013028&di=7562963243866521601&fr=click-pic&fromurl=http%253A%252F%252Fmbd.baidu.com%252Fnewspage%252Fdata%252Fdtlandingsuper%253Fnid%253Ddt_4094823392677133709&gsm=1e&hd=&height=0&hot=&ic=&ie=utf-8&imgformat=&imgratio=&imgspn=0&is=0%2C0&isImgSet=&latest=&lid=&lm=&objurl=https%253A%252F%252Fgips1.baidu.com%252Fit%252Fu%253D2151437926%252C3227013028%2526fm%253D3074%2526app%253D3074%2526f%253DJPEG&os=165877139%2C1077251114&pd=image_content&pi=0&pn=5&rn=1&simid=2151437926%2C3227013028&tn=baiduimagedetail&width=0&word=%E5%90%84%E7%A7%8D%E5%B0%8F%E5%8A%A8%E7%89%A9%E5%9B%BE%E7%89%87&z=
    # 小狗链接：https://image.baidu.com/search/detail?adpicid=0&b_applid=9292729119795015855&bdtype=0&commodity=&copyright=&cs=1324272466%2C1543020226&di=7562963243866521601&fr=click-pic&fromurl=http%253A%252F%252Fwww.douyin.com%252Fnote%252F7505407404030758156&gsm=78&hd=&height=0&hot=&ic=&ie=utf-8&imgformat=&imgratio=&imgspn=2&is=4042812812%2C1973328956&isImgSet=&latest=&lid=&lm=&objurl=https%253A%252F%252Fp3-pc-sign.douyinpic.com%252Ftos-cn-i-0813%252Fock9mKEBxAECfRDJADBPIjgIDfAYFAAMQCAoJQ~tplv-dy-aweme-images%253Aq75.webp&os=4042812812%2C1973328956&pd=image_content&pi=0&pn=94&rn=1&simid=1324272466%2C1543020226&tn=baiduimagedetail&width=0&word=%E5%B0%8F%E7%8B%97%E5%9B%BE%E7%89%87&z=