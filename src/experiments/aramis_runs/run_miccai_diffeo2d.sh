#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'diffeomorphism_atlas_2dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'


# ---------- run Bayesian 2d BraTs

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 1 --sliced_dim 2 \
--latent_dimension__s 10 --kernel_width__s 5 \
--lambda_square__s 1. --noise_variance 0.0025 --dropout .2 \
--epochs 10000 --batch_size 8 --nb_train 335 --nb_test 16 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 2 --lr .0005 --step_lr 3000 --update_from_epoch -1 \
--checkpoint_period 5 --write_every_epoch 25 --row_log_interval 2 --which_print "train" \
