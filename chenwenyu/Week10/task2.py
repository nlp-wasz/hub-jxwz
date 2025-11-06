from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

def extract_text_from_image(image_path):
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "../../../models/Qwen/Qwen2-VL-7B-Instruct",
        dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("../../../models/Qwen/Qwen2-VL-7B-Instruct")
    
    image = Image.open(image_path)
    
    # 使用明确的文字提取指令
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "请提取图片中的所有文字内容，按原样输出，不要添加任何解释。"}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
    output_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return output_text

# 使用示例
text_content = extract_text_from_image("./pics/text.jpg")
print("提取的文字:", text_content)
