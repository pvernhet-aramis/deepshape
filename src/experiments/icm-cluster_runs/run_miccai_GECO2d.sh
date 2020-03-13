#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=.
#SBATCH --output=outputs/scratch_baseline_job_%j.out
#SBATCH --error=outputs/scratch_baseline_job_%j.err
#SBATCH --job-name=ICM2dBraTsGECO
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
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesianGECO_atlas_2dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'

source /network/lustre/dtlake01/aramis/users/paul.vernhet/anaconda3/etc/profile.d/conda.sh
conda activate lightning_01

# ---------- run Bayesian 2d BraTs -- GECO version

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 1 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 10 --kernel_width__s 5 --kernel_width__a 2 \
--lambda_square__s 2. --lambda_square__a 1. --noise_variance 0.0025 --dropout .1 \
--epochs 20000 --batch_size 8 --nb_train 335 --nb_test 16 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 2 --lr .0005 --step_lr 3000 --update_every_batch 4 \
--checkpoint_period 25 --write_every_epoch 25 --row_log_interval 2 --which_print "train" \