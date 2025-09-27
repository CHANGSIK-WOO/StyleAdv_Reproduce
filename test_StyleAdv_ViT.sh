#!/usr/bin/env bash

#SBATCH --job-name=safe_test_ViT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=20G
#SBATCH --time=1-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err

set -euo pipefail

echo "Starting safe testing StyleAdv_ViT"
echo "Current working directory: $(pwd)"

# 모델 파일 존재 확인
if [ ! -f "output/ViT_5WAY_1SHOTS/best.pth" ]; then
    echo "Error: Model file not found. Please check if training is completed."
    exit 1
fi

# 안전한 테스트 실행
python test_StyleAdv_ViT.py

echo "Finished safe testing StyleAdv_ViT"