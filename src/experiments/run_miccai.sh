#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_3dmiccai.py'
OUTPUTDIR=${HOME}/'Results/MICCAI'
DATADIR=${HOME}/'Data/MICCAI_dataset'

# run example
python "${PYTHON_SCRIPT}" --output_dir "${OUTPUTDIR}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--downsampling_data 1 --downsampling_grid 1 \
--epochs 5000 --batch_size 5 --nb_train 150 --nb_test 10