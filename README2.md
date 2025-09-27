# StyleAdv Reproduction — BSCD-FSL (RN10 & ViT-Small)

This package assumes you already have the **StyleAdv-CDFSL** repo (uploaded) and a working GPU environment.

## 0. Conda env (baseline per upstream README)
```bash
conda create -n py36 python=3.6 -y
conda activate py36
conda install pytorch torchvision -c pytorch -y
conda install pandas -y
pip install 'scipy>=1.3.2' 'tensorboardX>=1.4' 'h5py>=2' tensorboard timm opencv-python==4.5.5.62 ml-collections
```

## 1. Prepare datasets (BS-CDFSL target sets)
- Prepare **ChestX**, **ISIC**, **EuroSAT**, **CropDisease** following IBM BS-CDFSL and the repo's README.
- Set paths in `StyleAdv-CDFSL/config_bscdfsl_dir.py`.
- Source (meta-train) dataset is **miniImageNet** as in the paper.

## 2. Reproduction runs
From the root of the uploaded repo (e.g., `StyleAdv-CDFSL/`), run the scripts in `repro_scripts/` created here.

```bash
# RN10, 5-way 1-shot (no FT)
bash repro_scripts/run_rn10_1shot.sh

# RN10, 5-way 1-shot (with FT on each target)
bash repro_scripts/run_rn10_1shot_ft.sh

# RN10, 5-way 5-shot (no FT)
bash repro_scripts/run_rn10_5shot.sh

# RN10, 5-way 5-shot (with FT on each target)
bash repro_scripts/run_rn10_5shot_ft.sh

# ViT-small (DINO/IN1K), 1-shot / 5-shot (no FT & FT)
bash repro_scripts/run_vits_1shot.sh
bash repro_scripts/run_vits_1shot_ft.sh
bash repro_scripts/run_vits_5shot.sh
bash repro_scripts/run_vits_5shot_ft.sh
```

- During meta-train, the repo **automatically evaluates** on BSCD-FSL. Acc files are saved under `output/checkpoints/<exp-name>/acc_*.txt`.
- FT (fine-tune) scripts resume the best checkpoint and run dataset-wise FT then test.

## 3. Expected numbers
Use the assignment's allowed margin (±5%). Fill your measured numbers into the LaTeX table included in `reports/reproduction.tex`.
(You can also directly paste into Word if you prefer; a `.docx` is provided.)

## 4. Idea proposal (Adaptive Support-Conditioned Style, AdaSCS)
- Code scaffold is under `ideas/adascs/` (lightweight, opt-in, minimally invasive).
- Turn it on with `--adascs` (RN) or `--adascs` (ViT) and a strength mode (e.g., `--adascs_mode=percentile`).
- See `ideas/adascs/README.md` and `ideas/adascs/patch_notes.txt` for usage & integration details.
- Ablation script templates are under `ideas/adascs/ablations/`.
