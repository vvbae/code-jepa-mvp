FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

COPY --from=uv_bin /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git libgl1-mesa-glx libglib2.0-0 ffmpeg build-essential curl tmux ncurses-term \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

RUN uv pip install --system \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0"

RUN uv pip install --system \
    torch_geometric \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

RUN uv pip install --system \
    "huggingface-hub>=0.23.0" \
    "transformers" \
    "datasets" \
    "accelerate" \
    "einops>=0.8.1" \
    "fire>=0.7.0" \
    "matplotlib>=3.10.3" \
    "opencv-python>=4.12.0.88" \
    "pudb>=2025.1" \
    "scikit-learn>=1.5.0" \
    "tiktoken>=0.9.0" \
    "torchcodec>=0.4.0" \
    "wandb[media]>=0.21.1" \
    "gymnasium>=1.1.1" \
    "imageio" \
    "seaborn" \
    "submitit" \
    "tqdm" \
    "pymunk" \
    "decord" \
    "omegaconf" \
    "tree-sitter" \
    "tree-sitter-python" \
    "ruamel.yaml"

RUN git config --global user.name "vvbae" && \
    git config --global user.email "polarsatellitest@gmail.com"

WORKDIR /workspace