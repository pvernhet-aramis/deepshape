#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesianGECO_atlas_3dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--downsampling_data 1 --downsampling_grid 2 \
--epochs 1000 --batch_size 2 --nb_train 256 --nb_test 8 --num_workers 0 \
--optimizer "AdaBound" --accumulate_grad_batch 12 --lr 0.005 --update_every_batch 2 --step_lr 100 \
--write_every_epoch 10 --row_log_interval 2 \
