#!/usr/bin/env bash

#SBATCH --job-name=metatrain_StyleAdv_RN_FT_5way_5shot
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

echo "Starting Sbatch metatrain_StyleAdv_RN_FT_5way_5shot"
# ISIC 데이터셋
python3 finetune_StyleAdv_RN.py --testset ISIC --name exp-FT-ISIC --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 0.001 --resume_dir exp-name/RN_5WAY_5SHOTS --resume_epoch -1

# EuroSAT 데이터셋
python3 finetune_StyleAdv_RN.py --testset EuroSAT --name exp-FT-EuroSAT --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 0.001 --resume_dir exp-name/RN_5WAY_5SHOTS --resume_epoch -1

# CropDisease 데이터셋
python3 finetune_StyleAdv_RN.py --testset CropDisease --name exp-FT-CropDisease --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 0.001 --resume_dir exp-name/RN_5WAY_5SHOTS --resume_epoch -1

# ChestX 데이터셋
python3 finetune_StyleAdv_RN.py --testset ChestX --name exp-FT-ChestX --train_aug --n_shot 5 --finetune_epoch 50 --finetune_LR 0.001 --resume_dir exp-name/RN_5WAY_5SHOTS --resume_epoch -1
echo "Finish Sbatch metatrain_StyleAdv_RN_FT_5way_5shot"