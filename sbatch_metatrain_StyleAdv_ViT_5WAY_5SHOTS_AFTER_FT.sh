#!/usr/bin/env bash

#SBATCH --job-name=sbatch_metatrain_StyleAdv_ViT_5WAY_5SHOTS_AFTER_FT
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

echo "Starting Sbatch sbatch_metatrain_StyleAdv_ViT_5WAY_5SHOTS_AFTER_FT"
# ISIC 데이터셋
python3 finetune_StyleAdv_ViT.py --testset ISIC --name exp-FT-ISIC --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 5e-5 --resume_dir ViT_5WAY_5SHOTS --resume_epoch -1

# EuroSAT 데이터셋
python3 finetune_StyleAdv_ViT.py --testset EuroSAT --name exp-FT-EuroSAT --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 5e-5 --resume_dir ViT_5WAY_5SHOTS --resume_epoch -1

# CropDisease 데이터셋
python3 finetune_StyleAdv_ViT.py --testset CropDisease --name exp-FT-CropDisease --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 5e-5 --resume_dir ViT_5WAY_5SHOTS --resume_epoch -1

# ChestX 데이터셋
python3 finetune_StyleAdv_ViT.py --testset ChestX --name exp-FT-ChestX --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 5e-5 --resume_dir ViT_5WAY_5SHOTS --resume_epoch -1
echo "Finish Sbatch sbatch_metatrain_StyleAdv_ViT_5WAY_5SHOTS_AFTER_FT"