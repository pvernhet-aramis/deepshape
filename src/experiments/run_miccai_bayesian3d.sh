#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_3dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'

# run overfitting example

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--downsampling_data 2 --downsampling_grid 2 \
--epochs 2000 --batch_size 1 --nb_train 1 --nb_test 1 --num_workers 0 \
--optimizer "AdaBound" --accumulate_grad_batch 1 --lr .001 --step_lr 50 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 2 \

#python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
#--cuda --num_gpu 0 \
#--downsampling_data 2 --downsampling_grid 2 \
#--epochs 2000 --batch_size 8 --nb_train 128 --nb_test 8 --num_workers 0 \
#--optimizer "AdaBound" --accumulate_grad_batch 4 --lr 0.0005 --step_lr 250 --update_from_epoch 500 \
#--write_every_epoch 25 --row_log_interval 2 \

#python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
#--cuda --num_gpu 0 \
#--downsampling_data 1 --downsampling_grid 2 \
#--epochs 2000 --batch_size 2 --nb_train 128 --nb_test 8 --num_workers 0 \
#--optimizer "AdaBound" --accumulate_grad_batch 8 --lr 0.0005 --step_lr 250 --update_from_epoch 500 \
#--write_every_epoch 25 --row_log_interval 2 \


