#!/bin/bash
#SBATCH --gpus=2

python experiments/stereoset/eap_ig_error_analysis.py --n-examples 100 --skip-validation --no-s3