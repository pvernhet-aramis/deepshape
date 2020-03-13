#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=.
#SBATCH --output=outputs/scratch_baseline_job_%j.out
#SBATCH --error=outputs/scratch_baseline_job_%j.err
#SBATCH --job-name=ICM3dBrats02
#SBATCH --gres=gpu:1
#SBATCH --mail-user=paul.vernhet@icm-institute.org
#SBATCH --mail-type=START,FAIL,END

# internet export
export http_proxy=http://10.10.2.1:8123
export https_proxy=http://10.10.2.1:8123

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_3dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'

source /network/lustre/dtlake01/aramis/users/paul.vernhet/anaconda3/etc/profile.d/conda.sh
conda activate lightning_01

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 2 --downsampling_grid 2 \
--latent_dimension__s 32 --latent_dimension__a 64 --kernel_width__s 10 --kernel_width__a 5 \
--lambda_square__s 1. --lambda_square__a 1. --noise_variance 0.0025 --dropout .1 \
--epochs 10000 --batch_size 2 --nb_train 335 --nb_test 16 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 8 --lr .001 --step_lr 150 --update_from_epoch -1 \
--checkpoint_period 5 --write_every_epoch 5 --row_log_interval 2 --which_print "train" \
