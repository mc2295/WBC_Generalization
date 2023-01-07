#!/bin/bash

#SBATCH --job-name=VGG16_augmented
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2500
#SBATCH --mail-type=NONE
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --nodelist=pas
eval "$(conda shell.bash hook)"     # Initialisation du shell pour conda
conda activate env2                # Activation de votre environnement python
/home/manon/.conda/envs/env2/bin/python /home/manon/codes/VGG16_augmented.py
