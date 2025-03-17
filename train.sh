#!/bin/sh
#SBATCH --job-name=training-gan
#SBATCH --partition=dgx-small
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thanhmaxdz2003@gmail.com
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Bắt đầu training trên GPU..."
nvidia-smi

module load cuda
module load cudnn
eval "$(conda shell.bash hook)"
conda activate thanhdz

python3 -u main.py  # -u để in log real-time
