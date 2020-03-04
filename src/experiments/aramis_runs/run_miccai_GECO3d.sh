#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesianGECO_atlas_3dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'

# run Bayesian 3d Mock

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--dataset "mock" --downsampling_data 1 --downsampling_grid 4 \
--latent_dimension__s 3 --latent_dimension__a 3 --kernel_width__s 2 --kernel_width__a 2 \
--lambda_square__s .1 --lambda_square__a .1 --noise_variance 0.0025 \
--epochs 500 --batch_size 4 --nb_train 1000 --nb_test 124 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 8 --lr .001 --step_lr 75 --update_from_epoch 250 --update_every_batch 64 \
--write_every_epoch 5 --row_log_interval 2 \

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--dataset "mock" --downsampling_data 1 --downsampling_grid 2 \
--latent_dimension__s 3 --latent_dimension__a 3 --kernel_width__s 5 --kernel_width__a 2 \
--lambda_square__s .1 --lambda_square__a .1 --noise_variance 0.0025 \
--epochs 500 --batch_size 4 --nb_train 1000 --nb_test 124 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 8 --lr .001 --step_lr 75 --update_from_epoch 250 --update_every_batch 64 \
--write_every_epoch 5 --row_log_interval 2 \

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--dataset "mock" --downsampling_data 1 --downsampling_grid 1 \
--latent_dimension__s 3 --latent_dimension__a 3 --kernel_width__s 5 --kernel_width__a 2 \
--lambda_square__s .1 --lambda_square__a .1 --noise_variance 0.0025 \
--epochs 500 --batch_size 4 --nb_train 1000 --nb_test 124 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 8 --lr .001 --step_lr 75 --update_from_epoch 250 --update_every_batch 64 \
--write_every_epoch 5 --row_log_interval 2 \
