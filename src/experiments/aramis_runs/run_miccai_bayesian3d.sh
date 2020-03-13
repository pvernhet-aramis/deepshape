#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_3dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'


# run Bayesian 3d Mock


#python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
#--cuda --num_gpu 1 \
#--dataset "mock" --downsampling_data 1 --downsampling_grid 4 \
#--latent_dimension__s 3 --latent_dimension__a 3 --kernel_width__s 2 --kernel_width__a 2 \
#--lambda_square__s .1 --lambda_square__a .1 --noise_variance 0.0025 \
#--epochs 500 --batch_size 4 --nb_train 1000 --nb_test 124 --num_workers 0 \
#--optimizer "Adam" --accumulate_grad_batch 8 --lr .001 --step_lr 75 --update_from_epoch 250 \
#--write_every_epoch 5 --row_log_interval 2 \


python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 4 \
--latent_dimension__s 20 --latent_dimension__a 15 --kernel_width__s 5 --kernel_width__a 1.5 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
--epochs 10000 --batch_size 2 --nb_train 330 --nb_test 16 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 8 --lr .001 --step_lr 2000 --update_from_epoch 7500 \
--write_every_epoch 25 --row_log_interval 2 --which_print "train" \


python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 1 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 \
--latent_dimension__s 20 --latent_dimension__a 15 --kernel_width__s 5 --kernel_width__a 2 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
--epochs 10000 --batch_size 2 --nb_train 330 --nb_test 16 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 8 --lr .0001 --step_lr 4000 --update_from_epoch 9000 \
--write_every_epoch 25 --row_log_interval 2 --which_print "train" \

