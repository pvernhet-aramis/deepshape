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
--downsampling_data 1 --downsampling_power 1 \
--epochs 10 --batch_size 2 --nb_train 5 --nb_test 3