#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesianGECO_atlas_3dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'


python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--dataset "mock" --downsampling_data 2 --downsampling_grid 2 \
--latent_dimension__s 20 --latent_dimension__a 10 --kernel_width__s 10 --kernel_width__a 2 \
--lambda_square__s 1. --lambda_square__a 1. --noise_variance 0.0025 \
--epochs 2000 --batch_size 4 --nb_train 4096 --nb_test 256 --num_workers 0 \
--optimizer "AdaBound" --accumulate_grad_batch 2 --lr 0.00005 --update_every_batch 128 --step_lr 50 --update_from_epoch 100 \
--write_every_epoch 25 --row_log_interval 2 \
