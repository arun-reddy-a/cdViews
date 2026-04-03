#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gpus=h100-80:2
#SBATCH -A cis250206p
#SBATCH -J som_cdviews
#SBATCH -o som_cdviews_%j.out

set -x

cd /ocean/projects/cis250206p/aanugu/cdViews/scripts

source ~/som/bin/activate

python3 som_cdViews_inference.py --cfg_file ../cfgs/QA.yaml --dataset SQA --num-gpus 2
