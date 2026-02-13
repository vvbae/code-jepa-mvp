import os
from huggingface_hub import hf_hub_download

# 启用高速下载模式
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = "nihilityd/code-jepa" 
filename = "你的模型文件名.pt"

# 如果是私有仓库，需要提供 token
token = os.getenv("HF_TOKEN") 

local_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=filename,
    token=token,
    local_dir="./"  # 下载到当前目录
)
print(f"模型已下载至: {local_path}")