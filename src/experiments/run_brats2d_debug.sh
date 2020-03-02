#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_2dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'

# run Bayesian 2d BraTs [Quick-Debug mode]

# ----- testing various params

#python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
#--cuda --num_gpu 0 \
#--dataset "brats" --downsampling_data 2 --downsampling_grid 2 --sliced_dim 2 \
#--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 \
#--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
#--epochs 5000 --batch_size 8 --nb_train 330 --nb_test 16 --num_workers 0 \
#--optimizer "Adam" --accumulate_grad_batch 2 --lr .001 --step_lr 500 --update_from_epoch 4000 \
#--monitor_train 'monitored_training_loss' --monitor_val 'monitored_validation_loss' --save_top_k_train 1 --save_top_k_val 2 \
#--write_every_epoch 5 --row_log_interval 2 --which_print "both" \


python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "mock" --downsampling_data 2 --downsampling_grid 1 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
--epochs 100 --batch_size 8 --nb_train 100 --nb_test 10 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .001 --step_lr 10 --update_from_epoch 25 \
--monitor_train 'monitored_loss_train' --monitor_val 'monitored_loss_validation' --save_top_k_train 2 --save_top_k_val 2 \
--write_every_epoch 1 --row_log_interval 1 --which_print "both" \
