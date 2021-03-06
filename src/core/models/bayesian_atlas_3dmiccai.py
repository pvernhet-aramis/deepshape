### Base ###
import fnmatch
import math
import os
import sys
import argparse
import datetime
from copy import deepcopy
import itertools
from functools import reduce
from operator import mul

### Visualization ###
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### Core ###
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# Setting paths to directory roots | >> deepshape
parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent)
os.chdir(parent)
print('Setting root path to : {}'.format(parent))

### IMPORTS ###
from src.in_out.datasets_miccai import *
from src.support.nets_miccai_3d import MetamorphicAtlas
from src.support.base_miccai import *

parser = argparse.ArgumentParser(description='Bayesian 3D Atlas MICCAI 2020.')
# action parameters
parser.add_argument('--output_dir', type=str, default='./results', help='Output directory root.')
parser.add_argument('--data_dir', type=str, default='./', help='Data directory root.')
parser.add_argument('--cuda', action='store_true', help='Whether CUDA is available on GPUs.')
parser.add_argument('--num_gpu', type=int, default=0, help='Which GPU to run on.')
parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use if cuda not available')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
# Dataset parameters
parser.add_argument('--downsampling_data', type=int, default=2, choices=[0, 1, 2], help='2**downsampling of initial data.')
# Model parameters
parser.add_argument('--downsampling_grid', type=int, default=2, choices=[0, 1, 2], help='2**downsampling of grid.')
parser.add_argument('--initialize_template', action='store_true', help='Whether to initialize template with random image.')
# Optimization parameters
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to perform.')
parser.add_argument('--nb_train', type=int, default=8, help='Number of training data.')
parser.add_argument('--nb_test', type=int, default=1, help='Number of testing data.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size when processing data.')

args = parser.parse_args()

# CPU/GPU settings || random seeds
args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    print('>> GPU available.')
    DEVICE = torch.device('cuda')
    torch.cuda.set_device(args.num_gpu)
    torch.cuda.manual_seed(args.seed)
else:
    DEVICE = torch.device('cpu')
    print('>> CUDA is not available. Overridding with device = "cpu".')
    print('>> OMP_NUM_THREADS will be set to ' + str(args.num_threads))
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)


def cyclic_dataloader(loader, device):
    for data, _ in itertools.cycle(loader):
        yield data.to(device)


print('>> Conda env: ', os.environ['CONDA_DEFAULT_ENV'])


def plot_losses(train_store, test_store, test_interval, path):
    N = len(train_store['losses']['all'])
    x = np.arange(1, N+1)
    x_eval = x[[e == 1 or e % test_interval == 0 for e in x]]

    # Fig 1 : full loss
    fig, ax = plt.subplots()
    ax.semilogy(x, train_store['losses']['all'], '-.', label='train', c='k')
    ax.semilogy(x_eval, test_store['losses']['all'], '--o', c='k', label='test')
    ax.tick_params(axis='y', labelcolor='k')
    ax.set_xlabel('epochs')
    ax.set_ylabel('global loss', color='k')
    ax.legend()
    plt.title('Global loss during optimization')
    plt.savefig(path + '_global' + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Fig 2 : kl __s and kl __ a
    fig, ax = plt.subplots()
    ax.semilogy(x, train_store['losses']['kl_s'], '-.', c='r', label='kl_s train')
    ax.semilogy(x_eval, test_store['losses']['kl_s'], '--o', c='r', label='kl_s test')
    axtwinx = ax.twinx()
    axtwinx.semilogy(x, train_store['losses']['kl_a'], '-.', c='b', label='kl_a train')
    axtwinx.semilogy(x_eval, test_store['losses']['kl_a'], '--o', c='b', label='kl_a test')
    ax.tick_params(axis='y', labelcolor='r')
    ax.set_xlabel('epochs')
    ax.set_ylabel('KL shape', color='r')
    axtwinx.set_ylabel('KL appearance', color='b')
    ax.legend()
    axtwinx.legend()
    plt.title('KL loss during optimization')
    plt.savefig(path + '_kl' + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Fig 3 : data attachement
    fig, ax = plt.subplots()
    ax.semilogy(x, train_store['losses']['rec'], '-.', label='train', c='k')
    ax.semilogy(x_eval, test_store['losses']['rec'], '--o', c='k', label='test')
    ax.tick_params(axis='y', labelcolor='k')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Reconstruction', color='k')
    ax.legend()
    plt.title('Reconstruction loss during optimization')
    plt.savefig(path + '_rec' + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Fig 4 : ratio evolution
    fig, ax = plt.subplots()
    ax.plot(x, train_store['ss']['var_s'], '-.', c='b', label='shape')
    axtwinx = ax.twinx()
    axtwinx.plot(x, train_store['ss']['var_a'], '-.', c='g', label='appearance')
    ax.tick_params(axis='y', labelcolor='k')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Lambdas shape', color='b')
    axtwinx.set_ylabel('Lambdas appearance', color='g')
    ax.legend()
    axtwinx.legend()
    plt.title('Lambdas evolution during optimization')
    plt.savefig(path + '_lambda' + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':

    # ==================================================================================================================
    # GLOBAL VARIABLES
    # ==================================================================================================================

    HOME_PATH = '/network/lustre/dtlake01/aramis/users/paul.vernhet'
    dataset_name = '3dBraTs'
    dataset_dirname = str(args.downsampling_data) + '_reduction'
    assert 0 <= args.downsampling_data < 3, "Downsampling limited to 2**2=4"

    number_of_images_train = args.nb_train  # 335
    number_of_images_test = args.nb_test  # 10
    batch_size = min(args.batch_size, number_of_images_train)

    downsampling_data = 2**args.downsampling_data
    downsampling_grid = 2**args.downsampling_grid
    number_of_time_points = 5

    dimension = 3
    latent_dimension__s = 10
    latent_dimension__a = 5

    kernel_width__s = 5
    kernel_width__a = 2.5

    lambda_square__s = 10 ** 2
    lambda_square__a = 10 ** 2
    noise_variance = 0.1 ** 2

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================

    print('>> Run with the BraTs 3D dataset.')

    experiment_prefix = '3D_rdm_slice_normalization_{}'.format(dataset_dirname)
    data_nifti_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/2_datasets/2_t1ce_normalized')
    data_tensor_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/2_t1ce_normalized', dataset_dirname)
    output_dir = os.path.join(HOME_PATH, 'Results/MICCAI/', dataset_name, experiment_prefix)

    # DATASET LOADERS ------------------------------
    np_affine = np.load(file=os.path.join(data_tensor_path, 'train', 'affine.npy'))
    args.np_affine = np_affine.tolist()      # only to keep track in experiments summary ... | may/should be omitted
    dataset_train = T13DDataset(os.path.join(data_tensor_path, 'train'), number_of_images_train,
                                init_seed=args.seed, check_endswith='pt')
    dataset_train.set_transform(dataset_train.standardizer)
    train_loader = data_utils.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_test = T13DDataset(os.path.join(data_tensor_path, 'test'), number_of_images_test,
                               init_seed=args.seed, check_endswith='pt')
    dataset_test.set_transform(dataset_train.standardizer)
    test_loader = data_utils.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    # INITIALIZE TEMPLATE TO ZERO ------------------------------
    # | intensities_template = dataset_train.__getitem__(np.random.choice(len(dataset_train), 1)).unsqueeze(0)
    if args.initialize_template:
        intensities_template = dataset_train.__getitem__(int(np.random.choice(len(dataset_train), 1))).unsqueeze(0)
    else:
        intensities_template = torch.zeros(dataset_train.mean.size()).unsqueeze(0)    # nul because of standardization
    intensities_mean = dataset_train.mean.detach().clone()                            # gpu_numpy_detach()
    intensities_std = dataset_train.std.detach().clone()                              # gpu_numpy_detach()

    # OPTIMIZATION ------------------------------
    number_of_epochs = args.epochs      # 5000
    print_every_n_iters = 10            # 100
    save_every_n_iters = 50             # 500

    learning_rate = 1e-3
    learning_rate_ratio = 1
    lr_decay = .5
    lr_patience = 5
    min_lr = 5e-5

    assert number_of_time_points > 1
    print('>> Dataset loaded successfully\n')

    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================

    log = ''
    args.model_signature = str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
    args.snapshots_path = os.path.join(output_dir, '{}'.format(args.model_signature))
    if not os.path.exists(args.snapshots_path):
        os.makedirs(args.snapshots_path)
    print('\n>> Setting output directory to:\n', args.snapshots_path)

    with open(os.path.join(args.snapshots_path, 'args.json'), 'w') as f:
        args_wo_device = deepcopy(args.__dict__)
        args_wo_device.pop('device', None)
        json.dump(args_wo_device, f, indent=4, sort_keys=True)

    # args.device = device

    # ==================================================================================================================
    # BUILD MODEL
    # ==================================================================================================================

    model = MetamorphicAtlas(
        intensities_template, number_of_time_points, downsampling_data, downsampling_grid,
        latent_dimension__s, latent_dimension__a,
        kernel_width__s, kernel_width__a,
        initial_lambda_square__s=lambda_square__s, initial_lambda_square__a=lambda_square__a)

    noise_dimension = reduce(mul, model.grid_size)

    # Get data on GPUs | else CPUs
    model = model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer_without_template = Adam([
    #     {'params': model.encoder.parameters()},
    #     {'params': model.decoder__s.parameters()},
    #     {'params': model.decoder__a.parameters()}
    # ], lr=learning_rate)

    # lr_decay = .9 | lr_patience = 20 | min_lr = 5e-4
    scheduler = ReduceLROnPlateau(optimizer, factor=lr_decay, patience=lr_patience, min_lr=min_lr, verbose=True,
                                 threshold_mode='abs')

    # ==================================================================================================================
    # RUN TRAINING
    # ==================================================================================================================

    # statistics trackers
    train_store = {
        'losses': {
            'rec': [],
            'kl_s': [],
            'kl_a': [],
            'all': [],
        },
        'ss': {
            'mean_s': [],
            'mean_a': [],
            'var_s': [],
            'var_a': [],
        }
    }
    test_store = {
        'losses': {
            'rec': [],
            'kl_s': [],
            'kl_a': [],
            'all': [],
        }
    }

    for epoch in range(number_of_epochs + 1):
        # scheduler.step()

        # ==============================================================================================================
        # TRAINING STEP
        # ==============================================================================================================

        model.train()

        n_step_per_epoch = len(train_loader)

        epoch_train_attachment_loss = np.zeros(n_step_per_epoch)
        epoch_train_kullback_regularity_loss__s = np.zeros(n_step_per_epoch)
        epoch_train_kullback_regularity_loss__a = np.zeros(n_step_per_epoch)
        epoch_train_total_loss = np.zeros(n_step_per_epoch)

        epoch_train_ss_s_mean = np.zeros(n_step_per_epoch)
        epoch_train_ss_s_var = np.zeros(n_step_per_epoch)
        epoch_train_ss_a_mean = np.zeros(n_step_per_epoch)
        epoch_train_ss_a_var = np.zeros(n_step_per_epoch)

        for batch_idx, intensities in enumerate(train_loader):
            batch_target_intensities = intensities.to(DEVICE)
            bts = batch_target_intensities.size(0)
            space_size = reduce(mul, batch_target_intensities.size()[2:])
            per_voxel_per_batch = float(bts * space_size)

            # ENCODE, SAMPLE AND DECODE
            means__s, log_variances__s, means__a, log_variances__a = model.encode(batch_target_intensities)
            log_variances__s = torch.clamp(log_variances__s, -8, 2)
            log_variances__a = torch.clamp(log_variances__a, -8, 2)
            stds__s, stds__a = torch.exp(0.5 * log_variances__s), torch.exp(0.5 * log_variances__a)

            ss_s_mean = gpu_numpy_detach(torch.mean(means__s))
            ss_a_mean = gpu_numpy_detach(torch.mean(means__a))
            ss_s_var = gpu_numpy_detach(torch.mean(means__s ** 2 + stds__s ** 2))
            ss_a_var = gpu_numpy_detach(torch.mean(means__a ** 2 + stds__a ** 2))

            batch_latent__s = means__s + torch.zeros_like(means__s).normal_() * stds__s
            batch_latent__a = means__a + torch.zeros_like(means__a).normal_() * stds__a
            transformed_template = model(batch_latent__s, batch_latent__a)

            # LOSS AVERAGED BY VOXEL
            attachment_loss = torch.sum((transformed_template - batch_target_intensities) ** 2) / noise_variance
            train_attachment_loss = gpu_numpy_detach(attachment_loss)

            kullback_regularity_loss__s = torch.sum(
                (means__s.pow(2) + log_variances__s.exp()) / lambda_square__s - log_variances__s + np.log(
                    lambda_square__s))
            # kullback_regularity_loss__s = torch.sum(means__s.pow(2)) / lambda_square__s
            train_kullback_regularity_loss__s = gpu_numpy_detach(kullback_regularity_loss__s)

            kullback_regularity_loss__a = torch.sum(
                (means__a.pow(2) + log_variances__a.exp()) / lambda_square__a - log_variances__a + np.log(
                    lambda_square__a))
            # kullback_regularity_loss__a = torch.sum(means__a.pow(2)) / lambda_square__a
            train_kullback_regularity_loss__a = gpu_numpy_detach(kullback_regularity_loss__a)

            total_loss = (attachment_loss + kullback_regularity_loss__s + kullback_regularity_loss__a) / per_voxel_per_batch
            # total_loss = attachment_loss
            train_total_loss = gpu_numpy_detach(attachment_loss + kullback_regularity_loss__s + kullback_regularity_loss__a)

            # GRADIENT STEP
            optimizer.zero_grad()
            # optimizer_without_template.zero_grad()
            total_loss.backward()
            if epoch > int(number_of_epochs * 0.5):
                # optimizer_without_template.step()
                optimizer.step()
            else:
                # optimizer_without_template.step()
                optimizer.step()

            ##############
            ### UPDATE ###
            ##############

            epoch_train_attachment_loss[batch_idx] = train_attachment_loss / per_voxel_per_batch
            epoch_train_kullback_regularity_loss__s[batch_idx] = train_kullback_regularity_loss__s / per_voxel_per_batch
            epoch_train_kullback_regularity_loss__a[batch_idx] = train_kullback_regularity_loss__a / per_voxel_per_batch
            epoch_train_total_loss[batch_idx] = train_total_loss / per_voxel_per_batch
            epoch_train_ss_s_mean[batch_idx] = ss_s_mean
            epoch_train_ss_s_var[batch_idx] = ss_s_var
            epoch_train_ss_a_mean[batch_idx] = ss_a_mean
            epoch_train_ss_a_var[batch_idx] = ss_a_var

            # Updates of hyperparameters
            if epoch > min(100, int(number_of_epochs * 0.5)):
                noise_variance *= float(train_attachment_loss / float(noise_dimension) / float(bts))
                lambda_square__s = float(ss_s_var / float(bts))
                lambda_square__a = float(ss_a_var / float(bts))

        epoch_train_attachment_loss = np.mean(epoch_train_attachment_loss)
        epoch_train_kullback_regularity_loss__s = np.mean(epoch_train_kullback_regularity_loss__s)
        epoch_train_kullback_regularity_loss__a = np.mean(epoch_train_kullback_regularity_loss__a)
        epoch_train_total_loss = np.mean(epoch_train_total_loss)

        epoch_train_ss_s_mean = np.mean(epoch_train_ss_s_mean)
        epoch_train_ss_s_var = np.mean(epoch_train_ss_s_var)
        epoch_train_ss_a_mean = np.mean(epoch_train_ss_a_mean)
        epoch_train_ss_a_var = np.mean(epoch_train_ss_a_var)

        # Store results
        train_store['losses']['rec'] += [epoch_train_attachment_loss]
        train_store['losses']['kl_s'] += [epoch_train_kullback_regularity_loss__s]
        train_store['losses']['kl_a'] += [epoch_train_kullback_regularity_loss__a]
        train_store['losses']['all'] += [epoch_train_total_loss]
        train_store['ss']['mean_s'] += [epoch_train_ss_s_mean]
        train_store['ss']['mean_a'] += [epoch_train_ss_a_mean]
        train_store['ss']['var_s'] += [epoch_train_ss_s_var]
        train_store['ss']['var_a'] += [epoch_train_ss_a_var]

        # ==============================================================================================================
        # TEST STEP
        # ==============================================================================================================

        if number_of_images_test > 1 and epoch % print_every_n_iters == 0:

            n_step_per_epoch = len(test_loader)

            epoch_test_attachment_loss = np.zeros(n_step_per_epoch)
            epoch_test_kullback_regularity_loss__s = np.zeros(n_step_per_epoch)
            epoch_test_kullback_regularity_loss__a = np.zeros(n_step_per_epoch)
            epoch_test_total_loss = np.zeros(n_step_per_epoch)

            for batch_idx, intensities in enumerate(test_loader):
                batch_target_intensities = intensities.to(DEVICE)
                bts = batch_target_intensities.size(0)
                space_size = reduce(mul, batch_target_intensities.size()[2:])
                per_voxel_per_batch = float(bts * space_size)

                # ENCODE, SAMPLE AND DECODE
                means__s, log_variances__s, means__a, log_variances__a = model.encode(batch_target_intensities)
                stds__s, stds__a = torch.exp(0.5 * log_variances__s), torch.exp(0.5 * log_variances__a)

                ss_s_mean = gpu_numpy_detach(torch.mean(means__s))
                ss_s_var = gpu_numpy_detach(torch.mean(means__s ** 2 + stds__s ** 2))
                ss_a_mean = gpu_numpy_detach(torch.mean(means__a))
                ss_a_var = gpu_numpy_detach(torch.mean(means__a ** 2 + stds__a ** 2))

                batch_latent__s = means__s + torch.zeros_like(means__s).normal_() * stds__s
                batch_latent__a = means__a + torch.zeros_like(means__a).normal_() * stds__a
                transformed_template = model(batch_latent__s, batch_latent__a)

                # LOSS
                attachment_loss = torch.sum((transformed_template - batch_target_intensities) ** 2) / noise_variance
                test_attachment_loss = gpu_numpy_detach(attachment_loss)

                kullback_regularity_loss__s = torch.sum(
                    (means__s.pow(2) + log_variances__s.exp()) / lambda_square__s - log_variances__s + np.log(
                        lambda_square__s))
                test_kullback_regularity_loss__s = gpu_numpy_detach(kullback_regularity_loss__s)

                kullback_regularity_loss__a = torch.sum(
                    (means__a.pow(2) + log_variances__a.exp()) / lambda_square__a - log_variances__a + np.log(
                        lambda_square__a))
                test_kullback_regularity_loss__a = gpu_numpy_detach(kullback_regularity_loss__a)

                total_loss = (attachment_loss + kullback_regularity_loss__s + kullback_regularity_loss__a) / per_voxel_per_batch
                test_total_loss = gpu_numpy_detach(attachment_loss + kullback_regularity_loss__s + kullback_regularity_loss__a)

                epoch_test_attachment_loss[batch_idx] = test_attachment_loss / per_voxel_per_batch
                epoch_test_kullback_regularity_loss__s[batch_idx] = test_kullback_regularity_loss__s / per_voxel_per_batch
                epoch_test_kullback_regularity_loss__a[batch_idx] = test_kullback_regularity_loss__a / per_voxel_per_batch
                epoch_test_total_loss[batch_idx] = test_total_loss / per_voxel_per_batch

            epoch_test_attachment_loss = np.mean(epoch_test_attachment_loss)
            epoch_test_kullback_regularity_loss__s = np.mean(epoch_test_kullback_regularity_loss__s)
            epoch_test_kullback_regularity_loss__a = np.mean(epoch_test_kullback_regularity_loss__a)
            epoch_test_total_loss = np.mean(epoch_test_total_loss)

            # Store results
            test_store['losses']['rec'] += [epoch_test_attachment_loss]
            test_store['losses']['kl_s'] += [epoch_test_kullback_regularity_loss__s]
            test_store['losses']['kl_a'] += [epoch_test_kullback_regularity_loss__a]
            test_store['losses']['all'] += [epoch_test_total_loss]

            scheduler.step(epoch_test_total_loss)

            # ==============================================================================================================
            # TEMPLATE SAFETY CHECK
            # ==============================================================================================================

            template_intensities = model.template_intensities.detach().clone()    # model.template_intensities.view((1,) + model.template_intensities.size())
            template_latent_s, _, template_latent_a, _ = model.encode(template_intensities)
            template_latent_s_norm = float(gpu_numpy_detach(torch.norm(template_latent_s[0], p=2)))
            template_latent_a_norm = float(gpu_numpy_detach(torch.norm(template_latent_a[0], p=2)))

        # ==============================================================================================================
        # WRITE IN LOG | SAVE MODEL
        # ==============================================================================================================

        if epoch % print_every_n_iters == 0 or epoch == number_of_epochs:
            log += cprint(
                '\n[Epoch: %d] Learning rate = %.2E ; Noise std = %.2E ; Template latent [ s ; a ] norms = [ %.3f ; %.3f ]'
                '\nss_s_mean = %.2E ; ss_s_var = %.2E ; lambda__s = %.2E'
                '\nss_a_mean = %.2E ; ss_a_var = %.2E ; lambda__a = %.2E'
                '\nTrain loss = %.3f (attachment = %.3f ; shape regularity = %.3f ; appearance regularity = %.3f)'
                '\nTest  loss = %.3f (attachment = %.3f ; shape regularity = %.3f ; appearance regularity = %.3f)' %
                (epoch, list(optimizer.param_groups)[0]['lr'], math.sqrt(noise_variance), template_latent_s_norm,
                 template_latent_a_norm,
                 epoch_train_ss_s_mean, epoch_train_ss_s_var, np.sqrt(lambda_square__s),
                 epoch_train_ss_a_mean, epoch_train_ss_a_var, np.sqrt(lambda_square__a),
                 epoch_train_total_loss, epoch_train_attachment_loss, epoch_train_kullback_regularity_loss__s,
                 epoch_train_kullback_regularity_loss__a,
                 epoch_test_total_loss, epoch_test_attachment_loss, epoch_test_kullback_regularity_loss__s,
                 epoch_test_kullback_regularity_loss__a))

            # Save losses plots
            plot_losses(train_store, test_store, test_interval=print_every_n_iters,
                        path=os.path.join(args.snapshots_path, 'plots'))

        if epoch % save_every_n_iters == 0 or epoch == number_of_epochs:
            print('>> Saving models and training samples ...')
            with open(os.path.join(args.snapshots_path, 'log.txt'), 'w') as f:
                f.write(log)

            torch.save(model.state_dict(), os.path.join(args.snapshots_path, 'model__epoch_%d.pth' % epoch))
            np.save(os.path.join(args.snapshots_path, 'vsa__epoch_%d' % epoch), gpu_numpy_detach(model.v_star_average))
            np.save(os.path.join(args.snapshots_path, 'nsa__epoch_%d' % epoch), gpu_numpy_detach(model.n_star_average))

            # Randomly select images
            n = min(5, number_of_images_test)
            intensities_to_write = []
            for batch_idx, intensities in enumerate(test_loader):
                if n <= 0:
                    break
                bts = intensities.size(0)
                nb_selected = min(bts, n)
                intensities_to_write.append(intensities[:nb_selected])
                n = n - nb_selected
            intensities_to_write = torch.cat(intensities_to_write).to(DEVICE)
            model.write(intensities_to_write, os.path.join(args.snapshots_path, 'train__epoch_%d' % epoch),
                        intensities_mean, intensities_std, affine=np_affine)
            print('>> Saving done')

