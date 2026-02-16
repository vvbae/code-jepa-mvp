#!/bin/bash

# --- 配置区 ---
DOCKER_USER="nihilityd"
IMAGE_NAME="code-jepa-mvp"
TAG="v1.0" # 每次更新可以改这个版本号，比如 v1.1, v1.2
# --------------

FULL_IMAGE_NAME="$DOCKER_USER/$IMAGE_NAME:$TAG"

echo "🚀 开始构建镜像: $FULL_IMAGE_NAME..."

# 构建镜像 (适配 Mac/Windows/Linux 的常用配置)
docker build -t $FULL_IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "✅ 构建成功！准备推送到 Docker Hub..."
    docker push $FULL_IMAGE_NAME
    if [ $? -eq 0 ]; then
        echo "🎉 推送成功！"
        echo "现在你可以在 RunPod 中使用镜像: $FULL_IMAGE_NAME"
    else
        echo "❌ 推送失败，请检查是否已执行 docker login"
    fi
else
    echo "❌ 构建失败，请检查 Dockerfile 语法"
fi