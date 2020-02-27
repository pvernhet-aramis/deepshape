#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_2dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'

# run Bayesian 2d Mock

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "mock" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 0 \
--latent_dimension__s 2 --latent_dimension__a 2 --kernel_width__s 10 --kernel_width__a 2 \
--lambda_square__s 0.01 --lambda_square__a 0.25 --noise_variance 0.0025 \
--epochs 500 --batch_size 32 --nb_train 10000 --nb_test 124 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 2 --lr .0005 --step_lr 50 --update_from_epoch 75 \
--write_every_epoch 5 --row_log_interval 2 \