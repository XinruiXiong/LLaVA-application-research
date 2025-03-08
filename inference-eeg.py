#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import threading
from PIL import Image

# ===== 导入 LLaVA 相关模块 =====
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token

# Hugging Face 里的流式输出工具
from transformers.generation.streamers import TextIteratorStreamer

def load_image(image_path: str) -> Image.Image:
    """
    加载并转换图像为 RGB 格式。
    如果传入的是 http/https 链接，会自动下载。
    否则直接读取本地文件。
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        import requests
        from io import BytesIO
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

def inference_eeg_image(
    model_path: str,
    base_model_name: str,
    image_path: str,
    prompt: str = "Please describe this EEG image",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 1.0
) -> str:
    """
    在单卡 GPU 上加载微调的 LLaVA (LoRA) 模型，对一张 EEG 图像做推理。
    :param model_path: 本地 LoRA 权重目录，如 "./checkpoints/llava-v1.5-13b-eeg-lora"
    :param base_model_name: 对应的基础模型名称，如 "lmsys/vicuna-13b-v1.5"
    :param image_path: EEG 图像路径（本地或 URL）
    :param prompt: 用户文本提示
    :param max_new_tokens: 生成的最大 token 数
    :param temperature: 采样温度
    :param top_p: top-p 采样阈值
    :return: 模型生成的文本字符串
    """
    # 1. 禁用冗余初始化
    disable_torch_init()
    print("Loading LLaVA from base model...")

    # 2. 加载模型 & tokenizer & image_processor
    #    这里用 device="cuda:0" 而非 device_map
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_name=os.path.basename(model_path),
        model_base=base_model_name,         # 底座模型名称，与微调时保持一致
        device="cuda:0",                    # 强制单卡
        torch_dtype=torch.float16,
        load_8bit=False,
        load_4bit=False
    )
    print("Model is loaded...")

    # 3. 读取并预处理图像
    image_data = load_image(image_path)
    image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values']
    # 原来：image_tensor = image_tensor.half().cuda()
    image_tensor = image_tensor.half().to("cuda:0")

    # 4. 构造多轮对话（这里只做一轮）
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    # 插入 <image> 占位符和用户 prompt
    inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # 5. 将 <image> token 替换并转为 input_ids
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()

    # 6. Streamer 用于流式输出，可改成一次性输出
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    # 7. 调用 model.generate 生成文本
    outputs = []
    def generation_thread():
        model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=False,        # 若想随机采样，可设 True
            temperature=temperature, # do_sample=False 时此参数不生效
            top_p=top_p,            # 同上
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True
        )

    thread = threading.Thread(target=generation_thread)
    thread.start()
    for new_text in streamer:
        outputs.append(new_text)
    thread.join()

    # 8. 拼接输出并去掉 stop_str
    raw_text = "".join(outputs)
    if raw_text.endswith(stop_str):
        raw_text = raw_text[:-len(stop_str)].rstrip()

    return raw_text

if __name__ == "__main__":
    # 示例：加载 LoRA 权重 + vicuna 底座，对本地 EEG 图像推理
    finetuned_model_path = "./checkpoints/llava-v1.5-13b-eeg-lora"
    base_model_name = "lmsys/vicuna-13b-v1.5"  # 微调时的底座模型

    # 示例 EEG 图像文件
    test_image_path = "./playground/data/images/test/n11939491/n11939491_1823_spectro_7306.JPEG"
    user_prompt = "Please describe this EEG image"

    result_text = inference_eeg_image(
        model_path=finetuned_model_path,
        base_model_name=base_model_name,
        image_path=test_image_path,
        prompt=user_prompt,
        max_new_tokens=256,
        temperature=0.2,
        top_p=1.0
    )
    print("模型输出：", result_text)
