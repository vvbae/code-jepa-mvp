import os
from huggingface_hub import HfApi

# 从环境变量中读取 Token
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("请先运行 'export HF_TOKEN=你的token'")

api = HfApi(token=token)

# --- 配置区 ---
REPO_ID = "nihilityd/code-jepa" 
LOCAL_FILE = "/workspace/code-jepa-mvp/code_jepa/mbpp_graphs.pt" # 你的 .pt 文件路径
# --------------

print(f"📦 正在准备上传至 {REPO_ID}...")

try:
    # 自动断点续传上传大文件
    api.upload_file(
        path_or_fileobj=LOCAL_FILE,
        path_in_repo=os.path.basename(LOCAL_FILE),
        repo_id=REPO_ID,
    )
    print("✨ 上传成功！你可以去网页端看看了。")
except Exception as e:
    print(f"❌ 出错了: {e}")