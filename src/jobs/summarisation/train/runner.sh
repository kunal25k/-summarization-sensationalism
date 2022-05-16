#!/bin/bash

params=$1
model=$2

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate DL
cd /scratch/csp9835/summarization-senationalism/src

python summarization/eval.py --params $params --model_path $model
