#!/bin/bash

# main args
HOME='/network/lustre/dtlake01/aramis/users/paul.vernhet'
PROJECT_PATH='Scripts/Ongoing/MICCAI2020/deepshape'
MODELS_PATH='src/core/models'
PYTHON_SCRIPT=${HOME}/${PROJECT_PATH}/${MODELS_PATH}/'bayesian_atlas_2dmiccai.py'
DATADIR=${HOME}/'Data/MICCAI_dataset'


# -------------------- reference run

#python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
#--cuda --num_gpu 0 \
#--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
#--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 --dropout 0.1 \
#--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
#--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
#--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
#--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

# -------------------- Lambda ratio

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 --dropout 0.1 \
--lambda_square__s 10. --lambda_square__a .5 --noise_variance 0.0025 \
--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 --dropout 0.1 \
--lambda_square__s 10. --lambda_square__a 200. --noise_variance 0.0025 \
--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 --dropout 0.1 \
--lambda_square__s 100. --lambda_square__a 100. --noise_variance 0.0025 \
--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

# -------------------- Noise ratio

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 --dropout 0.1 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.025 \
--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 --dropout 0.1 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.00025 \
--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

# -------------------- Kernel size for shape

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 1. --dropout 0.1 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a .5 --dropout 0.1 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
--epochs 1000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 1000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

# -------------------- Longer run for initial values

python "${PYTHON_SCRIPT}" --data_dir "${DATADIR}" \
--cuda --num_gpu 0 \
--dataset "brats" --downsampling_data 1 --downsampling_grid 2 --sliced_dim 2 \
--latent_dimension__s 10 --latent_dimension__a 5 --kernel_width__s 5 --kernel_width__a 2 --dropout 0.1 \
--lambda_square__s 10. --lambda_square__a 10. --noise_variance 0.0025 \
--epochs 10000 --batch_size 4 --nb_train 32 --nb_test 1 --num_workers 0 \
--optimizer "Adam" --accumulate_grad_batch 3 --lr .0005 --step_lr 2000 --update_from_epoch -1 \
--write_every_epoch 25 --row_log_interval 100 --which_print "train" \

