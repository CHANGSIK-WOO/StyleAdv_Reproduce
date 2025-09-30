#!/usr/bin/env bash

#SBATCH --job-name=metatrain_StyleAdv_RN_5way_5shot
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

echo "Starting Sbatch metatrain_StyleAdv_RN_5way_5shot"
#python3 metatrain_StyleAdv_RN.py --dataset miniImagenet --name exp-name --train_aug --warmup baseline --n_shot 5 --stop_epoch 200
python3 metatrain_StyleAdv_RN.py --dataset miniImagenet --name assignment_2_5WAY_5SHOTS --train_aug --warmup baseline --n_shot 5 --stop_epoch 200 --lambda_gram 0.5
echo "Finish Sbatch metatrain_StyleAdv_RN_5way_5shot"