import os
# from huggingface_hub import snapshot_download

# os.environ["http_proxy"] = "http://127.0.0.1:1081"
# os.environ["https_proxy"] = "http://127.0.0.1:1081"

# import torch
#
# print("PyTorch版本:", torch.__version__)  # 例如 2.0.1
# print("CUDA版本:", torch.version.cuda)  # 例如 11.7（若无GPU则返回 None）

# snapshot_download(repo_id='flamehaze1115/Wonder3D_plus', local_dir="./ckpts")
# snapshot_download(repo_id='botp/stable-diffusion-v1-5', local_dir="./ckpts/stable-diffusion-v1-5")


# 指定镜像源 URL
# snapshot_download(
#     repo_id="botp/stable-diffusion-v1-5",
#     local_dir="runwayml/stable-diffusion-v1-5",
#     endpoint="https://hf-mirror.com",  # 国内镜像地址
# )

# snapshot_download(
#     repo_id="h94/IP-Adapter",
#     local_dir="h94/IP-Adapter",
#     endpoint="https://hf-mirror.com",  # 国内镜像地址
# )

from modelscope import snapshot_download

cache_dir = "E:/ComfyUI/models/"
model_id = "stabilityai/stable-diffusion-2-inpainting"
# 'Qwen/Qwen-Image-Edit'
# iic/cv_fft_inpainting_lama
# model_dir = snapshot_download('Qwen/Qwen-Image-Edit', cache_dir=cache_dir)

model_dir = snapshot_download(model_id, cache_dir=cache_dir)
