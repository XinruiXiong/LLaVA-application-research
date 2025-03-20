# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import json
# import csv
# import torch
# import threading
# from PIL import Image

# # ======== LLaVA 相关 ========
# from llava.constants import IMAGE_TOKEN_INDEX
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token

# # Hugging Face 里的流式输出工具
# from transformers.generation.streamers import TextIteratorStreamer


# def load_llava_model_once(
#     model_path: str,
#     base_model_name: str,
#     device: str = "cuda:0"
# ):
#     """
#     一次性加载 LLaVA (LoRA) 模型、tokenizer、image_processor，
#     避免重复加载导致的资源消耗或错误。
#     """
#     disable_torch_init()
#     print("[INFO] Loading LLaVA from base model...")

#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path,
#         model_name=os.path.basename(model_path),
#         model_base=base_model_name,
#         device=device,
#         torch_dtype=torch.float16,
#         load_8bit=False,
#         load_4bit=False
#     )

#     print("[INFO] Model is loaded!")
#     return tokenizer, model, image_processor


# def load_image(image_path: str) -> Image.Image:
#     """
#     加载并转换图像为 RGB 格式。
#     如果是 http/https 链接，会自动下载。
#     否则直接读取本地文件。
#     """
#     if image_path.startswith("http://") or image_path.startswith("https://"):
#         import requests
#         from io import BytesIO
#         response = requests.get(image_path)
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#     else:
#         image = Image.open(image_path).convert("RGB")
#     return image


# def single_eeg_inference(
#     tokenizer,
#     model,
#     image_processor,
#     image_path: str,
#     user_prompt: str,
#     max_new_tokens: int = 256,
#     temperature: float = 0.2,
#     top_p: float = 1.0,
#     device: str = "cuda:0"
# ) -> str:
#     """
#     对**单张 EEG 图像**执行推理。
#     保留 JSON 对话里自带的 `<image>`；使用 tokenizer_image_token 来处理多模态输入。
#     """

#     # 1. 如果图像文件不存在，返回报错信息
#     if not os.path.exists(image_path):
#         return f"[ERROR] Image file not found: {image_path}"

#     # 2. 读取图像并预处理
#     image_data = load_image(image_path)
#     image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values']
#     image_tensor = image_tensor.half().to(device)

#     # 3. 不再手动拼接 <image>；直接用 user_prompt 中已有的 `<image>`
#     final_prompt = user_prompt.strip()
#     if not final_prompt:
#         # 如果万一没拿到任何文本，就给个默认提示
#         final_prompt = "<image>\nPlease describe the EEG image."

#     # 4. 将 <image> 转为多模态 token
#     input_ids = tokenizer_image_token(
#         final_prompt,
#         tokenizer,
#         IMAGE_TOKEN_INDEX,
#         return_tensors='pt'
#     ).to(device)

#     # 如果发现只有 1 维，则需要 unsqueeze(0) 变成 [batch_size, seq_len]
#     if input_ids.dim() == 1:
#         input_ids = input_ids.unsqueeze(0)

#     # 调试信息
#     # 你可以根据需要注释或保留
#     # print(f"[DEBUG] input_ids.shape = {tuple(input_ids.shape)}")
#     # print(f"[DEBUG] image_tensor.shape = {tuple(image_tensor.shape)}")

#     # 5. 使用 streamer 做流式输出（可改成一次性生成）
#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
#     outputs = []

#     def generation_thread():
#         model.generate(
#             inputs=input_ids,
#             images=image_tensor,
#             do_sample=False,
#             temperature=temperature,
#             top_p=top_p,
#             max_new_tokens=max_new_tokens,
#             streamer=streamer,
#             use_cache=True
#         )

#     thread = threading.Thread(target=generation_thread)
#     thread.start()
#     for new_text in streamer:
#         outputs.append(new_text)
#     thread.join()

#     # 6. 去掉末尾的分隔符（若有）
#     #    例如 LLaVA 的对话分隔符是 conv.sep 或 conv.sep2
#     #    你可以酌情决定是否要去掉
#     raw_text = "".join(outputs)
#     # 如果想用 conv_templates，可以把 conv_mode=llava_v1，sep=...,sep2=... 拿来
#     # 这里简单起见，假设不去额外处理

#     return raw_text.strip()


# def main():
#     """
#     读取 JSON (顶层是一个数组，每条记录里包含 'image' 路径和 'conversations')，
#     对每条记录做推理，并将结果写入 CSV。
#     """

#     # 1. 模型路径与 JSON 路径
#     finetuned_model_path = "./checkpoints/llava-v1.5-13b-eeg-lora"
#     base_model_name = "lmsys/vicuna-13b-v1.5"
#     json_path = "./playground/data/my_llava_eeg_test.json"
#     output_csv = "my_llava_eeg_results.csv"

#     # 2. 加载模型（只加载一次）
#     device = "cuda:0"
#     tokenizer, model, image_processor = load_llava_model_once(
#         model_path=finetuned_model_path,
#         base_model_name=base_model_name,
#         device=device
#     )

#     # 3. 读取 JSON
#     if not os.path.exists(json_path):
#         print(f"[ERROR] JSON file not found: {json_path}")
#         return

#     with open(json_path, "r", encoding="utf-8") as f:
#         data_list = json.load(f)

#     # 4. 打开 CSV 准备写入
#     with open(output_csv, mode="w", encoding="utf-8", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["id", "image", "prompt_used", "model_output"])

#         # 5. 遍历 JSON 数组的每条记录
#         for item in data_list:
#             item_id = item.get("id", "")
#             image_rel_path = item.get("image", "")
#             conversations = item.get("conversations", [])

#             # 取第一个 "human" 消息作为用户 prompt
#             user_prompt = ""
#             for turn in conversations:
#                 if turn["from"] == "human":
#                     user_prompt = turn["value"]
#                     break

#             # 如果 JSON 里是 "images/test/xxx"，但真实文件可能在 "./playground/data/images/test/xxx"
#             image_full_path = os.path.join("./playground/data", image_rel_path)

#             # 进行推理
#             answer = single_eeg_inference(
#                 tokenizer=tokenizer,
#                 model=model,
#                 image_processor=image_processor,
#                 image_path=image_full_path,
#                 user_prompt=user_prompt,
#                 max_new_tokens=256,
#                 temperature=0.2,
#                 top_p=1.0,
#                 device=device
#             )

#             writer.writerow([item_id, image_rel_path, user_prompt, answer])

#     print(f"[INFO] Done. Results saved to {output_csv}")


# if __name__ == "__main__":
#     main()





#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import csv
import torch
import threading
from PIL import Image

# ======== LLaVA 相关 ========
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token

# Hugging Face 里的流式输出工具
from transformers.generation.streamers import TextIteratorStreamer


def load_llava_model_once(
    model_path: str,
    base_model_name: str,
    device: str = "cuda:0"
):
    """
    一次性加载 LLaVA (LoRA) 模型、tokenizer、image_processor，
    避免重复加载导致的资源消耗或错误。
    """
    disable_torch_init()
    print("[INFO] Loading LLaVA from base model...")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_name=os.path.basename(model_path),
        model_base=base_model_name,
        device=device,
        torch_dtype=torch.float16,
        load_8bit=False,
        load_4bit=False
    )

    print("[INFO] Model is loaded!")
    return tokenizer, model, image_processor


def load_image(image_path: str) -> Image.Image:
    """
    加载并转换图像为 RGB 格式。
    如果是 http/https 链接，会自动下载。
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


def single_eeg_inference(
    tokenizer,
    model,
    image_processor,
    image_path: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 1.0,
    device: str = "cuda:0"
) -> str:
    """
    对**单张 EEG 图像**执行推理。
    保留 JSON 对话里自带的 `<image>`；使用 tokenizer_image_token 来处理多模态输入。
    """

    # 1. 如果图像文件不存在，返回报错信息
    if not os.path.exists(image_path):
        return f"[ERROR] Image file not found: {image_path}"

    # 2. 读取图像并预处理
    image_data = load_image(image_path)
    image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values']
    image_tensor = image_tensor.half().to(device)

    # 3. 如果 JSON 里没有 prompt，就给个默认提示
    final_prompt = user_prompt.strip()
    if not final_prompt:
        final_prompt = "<image>\nPlease describe the EEG image."

    # 4. 将 <image> 转为多模态 token
    input_ids = tokenizer_image_token(
        final_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).to(device)

    # 如果只有一维，则需要变成 [batch_size, seq_len]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # 5. 使用 streamer 做流式输出（可改成一次性生成）
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    outputs = []

    def generation_thread():
        model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=False,        # 如需随机采样可设为 True
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True
        )

    thread = threading.Thread(target=generation_thread)
    thread.start()
    for new_text in streamer:
        outputs.append(new_text)
    thread.join()

    # 6. 不做额外截断，直接返回
    raw_text = "".join(outputs)
    return raw_text.strip()


def main():
    """
    读取 JSON (顶层是一个数组，每条记录里包含 'image' 路径和 'conversations')，
    对每条记录做推理，并将结果写入 CSV。
    新增: 在 CSV 中添加一列 expected_output (即 JSON 中 "gpt" 的对话).
    """

    # 1. 模型路径与 JSON 路径
    finetuned_model_path = "./checkpoints/llava-v1.5-13b-eeg-lora"
    base_model_name = "lmsys/vicuna-13b-v1.5"
    json_path = "./playground/data/my_llava_eeg_test.json"
    output_csv = "my_llava_eeg_results.csv"

    # 2. 加载模型（只加载一次）
    device = "cuda:0"
    tokenizer, model, image_processor = load_llava_model_once(
        model_path=finetuned_model_path,
        base_model_name=base_model_name,
        device=device
    )

    # 3. 读取 JSON
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # 4. 打开 CSV 准备写入
    with open(output_csv, mode="w", encoding="utf-8", newline="") as csvfile:
        # 增加一列 "expected_output"
        writer = csv.writer(csvfile)
        writer.writerow(["id", "image", "prompt_used", "expected_output", "model_output"])

        # 5. 遍历 JSON 数组的每条记录
        for item in data_list:
            item_id = item.get("id", "")
            image_rel_path = item.get("image", "")
            conversations = item.get("conversations", [])

            # 从 conversations 里提取 prompt (human) & expected_output (gpt)
            user_prompt = ""
            expected_answer = ""
            for turn in conversations:
                if turn["from"] == "human":
                    user_prompt = turn["value"]
                elif turn["from"] == "gpt":
                    expected_answer = turn["value"]

            # 如果 JSON 里是 "images/test/xxx"，但真实文件可能在 "./playground/data/images/test/xxx"
            # 可自行拼接
            image_full_path = os.path.join("./playground/data", image_rel_path)

            # 进行推理
            answer = single_eeg_inference(
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                image_path=image_full_path,
                user_prompt=user_prompt,
                max_new_tokens=256,
                temperature=0.2,
                top_p=1.0,
                device=device
            )

            # 写 CSV，一行
            # 注意把 user_prompt 里的换行符转义，以免 CSV 结构乱
            safe_prompt = user_prompt.replace("\n", "\\n")
            safe_expected = expected_answer.replace("\n", "\\n")

            writer.writerow([
                item_id,
                image_rel_path,
                safe_prompt,
                safe_expected,
                answer
            ])

    print(f"[INFO] Done. Results saved to {output_csv}")


if __name__ == "__main__":
    main()
