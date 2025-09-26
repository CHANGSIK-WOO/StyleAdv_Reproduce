#!/usr/bin/env bash

#SBATCH --job-name=metatrain_StyleAdv_RN_5way_1shot
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time=1-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err

set -euo pipefail

. /home/$USER/anaconda3/etc/profile.d/conda.sh
conda activate py36

echo "Starting Sbatch metatrain_StyleAdv_RN_5way_1shot"
python3 metatrain_StyleAdv_RN.py --dataset miniImagenet --name exp-name --train_aug --warmup baseline --n_shot 1 --stop_epoch 200
