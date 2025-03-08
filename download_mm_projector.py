from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5",
    filename="mm_projector.bin",
    local_dir="./checkpoints/llava-v1.5-13b-pretrain"
)
