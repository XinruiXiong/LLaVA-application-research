from huggingface_hub import HfApi

# repo_name = "Xinrui01/llava-13b-lora"
repo_name = "Xinrui01/llava-13b-eeg-lora"
# local_model_path = "./checkpoints/llava-v1.5-13b-lora"
local_model_path = "./checkpoints/llava-v1.5-13b-eeg-lora"

api = HfApi()
api.create_repo(repo_name, private=False)  # 设置为 True 变成私有仓库
api.upload_folder(folder_path=local_model_path, repo_id=repo_name)
