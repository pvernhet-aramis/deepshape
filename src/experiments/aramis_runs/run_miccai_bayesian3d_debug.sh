#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_3dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'


# run Bayesian 3d Mock

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 \
--lambda_square__s 5. --lambda_square__a 10. --noise_variance 0.0025 \
--epochs 10000 --batch_size 2 --nb_train 3 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 1 --lr .0005 --step_lr 3000 --update_from_epoch -1 \
--checkpoint_period 5 --write_every_epoch 5 --row_log_interval 2 --which_print "both" \

