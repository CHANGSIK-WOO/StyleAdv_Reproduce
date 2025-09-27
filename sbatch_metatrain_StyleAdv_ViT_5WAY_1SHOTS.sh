#!/usr/bin/env bash

#SBATCH --job-name=metatrain_StyleAdv_ViT_5way_1shot
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

echo "Starting Sbatch metatrain_StyleAdv_ViT_5way_1shot"
python metatrain_StyleAdv_ViT_with_test.py --output output/ViT_5WAY_1SHOTS --dataset mini_imagenet --epoch 20 --lr 5e-5 --arch dino_small_patch16 --device cuda --nSupport 1 --fp16
echo "Finish Sbatch metatrain_StyleAdv_ViT_5way_1shot"