#!/usr/bin/env bash

#SBATCH --job-name=sbatch_metatrain_StyleAdv_ViT_5WAY_5SHOTS_BEFORE_FT
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

echo "Starting Sbatch sbatch_metatrain_StyleAdv_ViT_5WAY_5SHOTS_BEFORE_FT.sh"
python metatrain_StyleAdv_ViT_Test_5WAY_5SHOTS.py --n_shot 5
echo "Finish Sbatch sbatch_metatrain_StyleAdv_ViT_5WAY_5SHOTS_BEFORE_FT.sh"
