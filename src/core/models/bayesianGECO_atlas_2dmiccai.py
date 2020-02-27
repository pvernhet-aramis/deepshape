import os
import sys
import argparse
import datetime
import math
from copy import deepcopy
import logging

### Visualization ###
import matplotlib
matplotlib.use('Agg')

### Core ###
import numpy as np
from torch.optim import Adam
from adabound import AdaBound
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl


# Setting paths to directory roots | >> deepshape
parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent)
os.chdir(parent)
print('Setting root path to : {}'.format(parent))

### IMPORTS ###
from src.in_out.datasets_miccai import ZeroOneT12DDataset
from src.support.networks.nets_vae_2d import MetamorphicAtlas2d
from src.support.base_miccai import *


# ---------------------------------------------------------------------


class VariationalMetamorphicAtlas2dExecuter(pl.LightningModule):

    def __init__(self, hparams, model):
        super(VariationalMetamorphicAtlas2dExecuter, self).__init__()
        self.hparams = hparams
        self.check_hparams()

        # nn.Module parameters
        self.model = model
        self.mse = torch.nn.MSELoss(reduction='sum')

        # Miscellenious holders
        self.model = model
        self.lambda_lagrangian = deepcopy(hparams.lambda_lagrangian)
        self.kappa = deepcopy(hparams.kappa)
        self.alpha_smoothing = deepcopy(hparams.alpha_smoothing)
        self.moving_avg = None
        self.last_device = None
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.previous_atlas = self.model.template_intensities.clone().detach()

        # Datasets
        dataset_train = ZeroOneT12DDataset(os.path.join(self.hparams.data_tensor_path, 'train'), self.hparams.nb_train,
                                           sliced_dim=self.hparams.sliced_dim,
                                           reduction=self.hparams.downsampling_data, init_seed=self.hparams.seed,
                                           check_endswith='pt', is_half=self.hparams.use_16bits)
        self.train_loader = data_utils.DataLoader(dataset_train, batch_size=self.hparams.batch_size, shuffle=True,
                                                  num_workers=self.hparams.num_workers,
                                                  pin_memory=self.hparams.pin_memory)
        dataset_val = ZeroOneT12DDataset(os.path.join(self.hparams.data_tensor_path, 'test'), self.hparams.nb_test,
                                         reduction=self.hparams.downsampling_data, init_seed=self.hparams.seed,
                                         sliced_dim=self.hparams.sliced_dim,
                                         check_endswith='pt', is_half=self.hparams.use_16bits)
        self.val_loader = data_utils.DataLoader(dataset_val, batch_size=self.hparams.batch_size, shuffle=True,
                                                num_workers=self.hparams.num_workers,
                                                pin_memory=self.hparams.pin_memory)

    def check_hparams(self):
        assert isinstance(self.hparams.num_workers, int) and self.hparams.num_workers >= 0, "num workers must be int"

    def moving_averager(self, current, previous, is_first=False):
        return current if is_first else self.alpha_smoothing * previous + (1-self.alpha_smoothing) * current

    def forward(self, x):
        return self.model.decode(x)

    def training_step(self, batch, batch_idx):
        """
        Variational Autoencoder step : KL divergence loss
        """
        self.last_device = batch.device.index
        batch_target_intensities = batch
        bts = float(batch_target_intensities.size(0))
        space_size = float(reduce(mul, batch_target_intensities.size()[2:]))

        # ---------- ENCODE, SAMPLE AND DECODE
        means__s, log_variances__s, means__a, log_variances__a = self.model.encode(batch_target_intensities)
        log_variances__s = torch.clamp(log_variances__s, self.hparams.clipvar_min, self.hparams.clipvar_max)
        log_variances__a = torch.clamp(log_variances__a, self.hparams.clipvar_min, self.hparams.clipvar_max)
        stds__s, stds__a = torch.exp(0.5 * log_variances__s), torch.exp(0.5 * log_variances__a)

        ss_s_mean = gpu_numpy_detach(torch.mean(means__s))
        ss_a_mean = gpu_numpy_detach(torch.mean(means__a))
        ss_s_var = gpu_numpy_detach(torch.mean(means__s ** 2 + stds__s ** 2))
        ss_a_var = gpu_numpy_detach(torch.mean(means__a ** 2 + stds__a ** 2))

        batch_latent__s = means__s + torch.zeros_like(means__s).normal_() * stds__s
        batch_latent__a = means__a + torch.zeros_like(means__a).normal_() * stds__a
        transformed_template = model(batch_latent__s, batch_latent__a)

        # ---------- LOSS AVERAGED BY VOXEL
        attachment_loss = self.mse(transformed_template, batch_target_intensities) / self.model.noise_variance
        kl_loss__s = torch.sum(
            (means__s.pow(2) + log_variances__s.exp()) / self.model.lambda_square__s - log_variances__s + np.log(
                self.model.lambda_square__s))
        kl_loss__a = torch.sum(
            (means__a.pow(2) + log_variances__a.exp()) / self.model.lambda_square__a - log_variances__a + np.log(
                self.model.lambda_square__a))

        constrain = (self.mse(transformed_template, batch_target_intensities) / bts - space_size * self.kappa ** 2) / self.model.noise_variance
        total_loss = self.lambda_lagrangian * constrain + (kl_loss__s + kl_loss__a) / bts

        # ---------- LOGS
        self.logger.experiment.add_scalars('attachment_loss', {'train': attachment_loss}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__s', {'train': kl_loss__s}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__a', {'train': kl_loss__a}, self.global_step)
        self.logger.experiment.add_scalars('lambda_lagrangian', {'train': self.lambda_lagrangian}, self.global_step)
        self.logger.experiment.add_scalars('total_loss', {'train': total_loss}, self.global_step)
        self.logger.experiment.add_scalars('ss_s_mean', {'train': ss_s_mean}, self.global_step)
        self.logger.experiment.add_scalars('ss_a_mean', {'train': ss_a_mean}, self.global_step)
        self.logger.experiment.add_scalars('ss_s_var', {'train': ss_s_var}, self.global_step)
        self.logger.experiment.add_scalars('ss_a_var', {'train': ss_a_var}, self.global_step)
        self.logger.experiment.add_scalar('lr', self.trainer.optimizers[0].param_groups[0]['lr'], self.global_step)

        # ---------- KEEP TRACKS FOR PARAMS CUSTOMIZED UPDATES
        self.moving_avg = float(
            gpu_numpy_detach(self.moving_averager(constrain, self.moving_avg, not self.global_step)))

        return {'loss': total_loss}

    def on_after_backward(self):
        """
        Hyper-parameters update | batch_level
        """
        # -------- LOG ATLAS VARIATIONS
        atlas_delta = gpu_numpy_detach(torch.sum(torch.abs(self.previous_atlas - self.model.template_intensities)))
        self.logger.experiment.add_scalar('atlas_total_L1_variations', atlas_delta, self.global_step)
        self.previous_atlas = self.model.template_intensities.clone().detach()

        # -------- UPDATE PARAMETERS IF NECESSARY
        if self.trainer.current_epoch and self.trainer.current_epoch % self.hparams.update_every_batch == 0:
            self.lambda_lagrangian *= float(np.clip(np.exp(self.moving_avg), 0.99, 1.01))

    def validation_step(self, batch, batch_idx):
        """
        Variational Autoencoder step : KL divergence
        """
        self.last_device = batch.device.index
        batch_target_intensities = batch
        bts = batch_target_intensities.size(0)
        space_size = reduce(mul, batch_target_intensities.size()[2:])

        # ---------- ENCODE, SAMPLE AND DECODE
        means__s, log_variances__s, means__a, log_variances__a = self.model.encode(batch_target_intensities)
        log_variances__s = torch.clamp(log_variances__s, self.hparams.clipvar_min, self.hparams.clipvar_max)
        log_variances__a = torch.clamp(log_variances__a, self.hparams.clipvar_min, self.hparams.clipvar_max)
        stds__s, stds__a = torch.exp(0.5 * log_variances__s), torch.exp(0.5 * log_variances__a)

        batch_latent__s = means__s + torch.zeros_like(means__s).normal_() * stds__s
        batch_latent__a = means__a + torch.zeros_like(means__a).normal_() * stds__a
        transformed_template = model(batch_latent__s, batch_latent__a)

        # ---------- LOSS AVERAGED BY VOXEL
        attachment_loss = self.mse(transformed_template, batch_target_intensities) / self.model.noise_variance
        kl_loss__s = torch.sum(
            (means__s.pow(2) + log_variances__s.exp()) / self.model.lambda_square__s - log_variances__s + np.log(
                self.model.lambda_square__s))
        kl_loss__a = torch.sum(
            (means__a.pow(2) + log_variances__a.exp()) / self.model.lambda_square__a - log_variances__a + np.log(
                self.model.lambda_square__a))

        constrain = (self.mse(transformed_template, batch_target_intensities) / bts - space_size * self.kappa ** 2) / self.model.noise_variance
        total_loss = self.lambda_lagrangian * constrain + (kl_loss__s + kl_loss__a) / bts

        outputs = {
            'val_attachment_loss': attachment_loss,
            'val_kl_loss__s': kl_loss__s,
            'val_kl_loss__a': kl_loss__a,
            'val_total_loss': total_loss,
        }
        return outputs

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        val_attachment_loss_mean = 0
        val_kl_loss__s_mean = 0
        val_kl_loss__a_mean = 0
        val_total_loss_mean = 0
        for output in outputs:
            val_attachment_loss_mean += output['val_attachment_loss']
            val_kl_loss__s_mean += output['val_kl_loss__s']
            val_kl_loss__a_mean += output['val_kl_loss__a']
            val_total_loss_mean += output['val_total_loss']

        val_attachment_loss_mean /= len(outputs)
        val_kl_loss__s_mean /= len(outputs)
        val_kl_loss__a_mean /= len(outputs)
        val_total_loss_mean /= len(outputs)

        self.logger.experiment.add_scalars('attachment_loss', {'val': val_attachment_loss_mean}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__s', {'val': val_kl_loss__s_mean}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__a', {'val': val_kl_loss__a_mean}, self.global_step)
        self.logger.experiment.add_scalars('total_loss', {'val': val_total_loss_mean}, self.global_step)

        return {'val_loss': val_total_loss_mean, 'progress_bar': {'val_loss': val_total_loss_mean}}

    def configure_optimizers(self):

        if self.hparams.optimizer.lower() == 'adam':
            print('>> Adam optimizer chosen !')
            base_opt = Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        elif self.hparams.optimizer.lower() == 'adabound':
            print('>> AdaBound optimizer chosen !')
            base_opt = AdaBound(self.model.parameters(),
                                lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2),
                                gamma=(1. - self.hparams.b2))
        else:
            assert False, 'No optimizer specified, please choose !'
        optimizer = [base_opt]
        scheduler = [StepLR(base_opt, self.hparams.step_lr, gamma=self.hparams.step_decay)]
        return optimizer, scheduler

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    @pl.data_loader
    def test_dataloader(self):
        return self.val_loader

    def on_epoch_end(self):
        """
        Called on end of training epoch | save viz and nifti according to current epoch value
        """
        if self.trainer.current_epoch == 0 or self.trainer.current_epoch % self.hparams.write_every_epoch == 0:
            self.save_viz()

    def save_model(self):
        """
        Save model (akin to checkpoints)
        """
        torch.save(self.model.state_dict(), os.path.join(self.hparams.snapshots_path,
                                                         'model__epoch_%d.pth' % self.current_epoch))

    def save_viz(self):
        """
        Saving nifti images
        """
        # Randomly select images
        test_loader = self.test_dataloader()     # by default, returned dataloaders are in list
        n = min(5, self.hparams.nb_test)
        intensities_to_write = []
        for batch_idx, intensities in enumerate(test_loader[0]):
            if n <= 0:
                break
            bts = intensities.size(0)
            nb_selected = min(bts, n)
            intensities_to_write.append(intensities[:nb_selected])
            n = n - nb_selected
        intensities_to_write = torch.cat(intensities_to_write)
        if self.on_gpu:
            intensities_to_write = intensities_to_write.cuda(self.last_device)
        if self.hparams.use_16bits:
            intensities_to_write = intensities_to_write.half()
        self.model.write(intensities_to_write, os.path.join(self.hparams.snapshots_path, 'train__epoch_%d' % self.current_epoch),
                         is_half=self.hparams.use_16bits)
        print('>> Saving done')


if __name__ == '__main__':

    # ==================================================================================================================
    # GLOBAL VARIABLES
    # ==================================================================================================================

    parser = argparse.ArgumentParser(description='Bayesian 2D Atlas MICCAI 2020 | LIGHTNING VERSION.')
    # action parameters
    parser.add_argument('--data_dir', type=str, default='Data/MICCAI_dataset',
                        help='Data directory root.')
    parser.add_argument('--cuda', action='store_true', help='Whether CUDA is available on GPUs.')
    parser.add_argument('--num_gpu', type=int, default=0, help='Which GPU to run on.')
    parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use if cuda not available')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='mock', choices=['mock', 'brats'],
                        help='Dataset choice between mock eyes and brats.')
    parser.add_argument('--sliced_dim', type=int, default=2, choices=[0, 1, 2],
                        help='Sliced dimension (without channel).')
    parser.add_argument('--dimension', type=int, default=2, choices=[2], help='Dataset dimension.')
    parser.add_argument('--downsampling_data', type=int, default=2**1, choices=[1, 2, 4],
                        help='2**downsampling of initial data.')
    # Model parameters
    parser.add_argument('--latent_dimension__s', type=int, default=10, help='Latent dimension of s.')
    parser.add_argument('--latent_dimension__a', type=int, default=5, help='Latent dimension of a.')
    parser.add_argument('--kernel_width__s', type=int, default=5, help='Kernel width s.')
    parser.add_argument('--kernel_width__a', type=int, default=2.5, help='Kernel width a.')
    parser.add_argument('--lambda_square__s', type=float, default=10. ** 2, help='Lambda square s.')
    parser.add_argument('--lambda_square__a', type=float, default=10. ** 2, help='Lambda square a.')
    parser.add_argument('--noise_variance', type=float, default=0.1 ** 2, help='Noise variance.')
    parser.add_argument('--downsampling_grid', type=int, default=2**1, choices=[1, 2, 4],
                        help='2**downsampling of grid.')
    parser.add_argument('--number_of_time_points', type=int, default=5, help='Integration time points.')
    # Training parameters
    parser.add_argument('--clipvar_min', type=float, default=float(-5*np.log(10)), help='10**min clip variance.')
    parser.add_argument('--clipvar_max', type=float, default=float(3*np.log(10)), help='10**max clip variance.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to perform.')
    parser.add_argument('--nb_train', type=int, default=32, help='Number of training data.')
    parser.add_argument('--nb_test', type=int, default=8, help='Number of testing data.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size when processing data.')
    parser.add_argument('--accumulate_grad_batch', type=int, default=2,
                        help='Number of accumulated batch for grad step.')
    parser.add_argument('--use_16bits', action='store_true', help='Whether to use 16-bits mixed precision.')
    parser.add_argument("--amp_level", type=str, default="O2", choices=["O0", "O1", "O2", "O3"],
                        help="automatic mixed precision level")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloaders.')
    parser.add_argument('--pin_memory', action='store_true', help='Whether to pin memory for dataloaders.')
    # Optimization parameters
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['Adam', 'AdaBound'],
                        help="Choose between Adam, or AdaBound")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam or AdaBound: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam or AdaBound: first order momentum decay")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam or AdaBound: second order momentum decay")
    parser.add_argument("--gamma", type=float, default=0.005, help="Adabound: convergence speed of bound functions")
    parser.add_argument('--lambda_lagrangian', type=float, default=1., help='Lagrange init. coefficient for GECO loss.')
    parser.add_argument('--kappa', type=float, default=float(np.sqrt(0.001)),
                        help='Kappa sensitivity hyperparameter for reconstruction loss.')
    parser.add_argument('--alpha_smoothing', type=float, default=0.99, help='GECO moving average loss.')
    parser.add_argument('--update_every_batch', type=int, default=2, help='When to update lambda.')
    parser.add_argument('--step_lr', type=int, default=500, help='learning rate scheduler every epoch activation.')
    parser.add_argument('--step_decay', type=float, default=.5, help='learning rate scheduler decay value.')
    # Storing data parameters
    parser.add_argument('--write_every_epoch', type=int, default=50, help='Number of iterations for checkpoints.')
    parser.add_argument('--row_log_interval', type=int, default=10, help='Log interval.')
    parser.add_argument('--track_norms', type=int, default=-1, help='Track gradients norms (default: None).')

    args = parser.parse_args()

    # GLOBAL
    HOME_PATH = '/network/lustre/dtlake01/aramis/users/paul.vernhet'

    # dataset-related args
    args.experiment_prefix = '2D_rdm_slice{}_normalization_{}_reduction'.format(args.sliced_dim, args.downsampling_data)
    args.data_tensor_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d',
                                         '2_t1ce_normalized/0_reduction' if args.dataset == 'brats' else '1_eyes')
    args.output_dir = os.path.join(HOME_PATH, 'Results/MICCAI',
                                   '2dBraTs' if args.dataset == 'brats' else '2dEyes', args.experiment_prefix)

    # batch-related args
    args.batch_size = min(args.batch_size, args.nb_train)

    assert args.nb_train >= args.batch_size * args.accumulate_grad_batch, \
        ('Incompatible options ( n_train = %d ) < ( batch_size * accumulate_grad_batches = %d * %d )' %
         (args.nb_train, args.batch_size, args.accumulate_grad_batch))

    # ==================================================================================================================
    # GPU SETUP | SEEDS
    # ==================================================================================================================

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

    # print('>> Conda env: ', os.environ['CONDA_DEFAULT_ENV'])

    # ==================================================================================================================
    # SNAPSHOTS | SETS SAVE DIRECTORIES
    # ==================================================================================================================

    log = ''
    args.model_signature = str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
    args.snapshots_path = os.path.join(args.output_dir, 'VAE_{}'.format(args.model_signature))
    if not os.path.exists(args.snapshots_path):
        os.makedirs(args.snapshots_path)
    print('\n>> Setting output directory to:\n', args.snapshots_path)

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================

    # INITIALIZE TEMPLATE TO MEAN OF TRAINING DATA ------------------------------
    dataset_train = ZeroOneT12DDataset(os.path.join(args.data_tensor_path, 'train'), args.nb_train,
                                       reduction=args.downsampling_data, sliced_dim=args.sliced_dim,
                                       init_seed=args.seed, check_endswith='pt')
    intensities_template, _ = dataset_train.compute_statistics()
    intensities_template = intensities_template.unsqueeze(0)         # (batch, channel, width, height)
    assert len(intensities_template.size()) == args.dimension + 2, "atlas size must be (batch, channel, width, height)"
    assert not torch.isnan(intensities_template).any(), "NaN detected"
    del dataset_train
    print('>> Templated initialized successfully\n')

    # ==================================================================================================================
    # BUILD MODEL
    # ==================================================================================================================

    model = MetamorphicAtlas2d(
        intensities_template, args.number_of_time_points, args.downsampling_data, args.downsampling_grid,
        args.latent_dimension__s, args.latent_dimension__a,
        args.kernel_width__s, args.kernel_width__a,
        initial_lambda_square__s=args.lambda_square__s, initial_lambda_square__a=args.lambda_square__a,
        noise_variance=args.noise_variance).to(DEVICE)

    # ==================================================================================================================
    # RUN TRAINING
    # ==================================================================================================================

    VAE_metamorphic = VariationalMetamorphicAtlas2dExecuter(args, model)

    custom_early_stop_callback = None

    trainer = pl.Trainer(gpus=([args.num_gpu] if args.cuda else None),
                         default_save_path=args.snapshots_path,
                         max_epochs=args.epochs,
                         early_stop_callback=custom_early_stop_callback,
                         use_amp=args.use_16bits,
                         amp_level=args.amp_level,
                         track_grad_norm=args.track_norms,
                         accumulate_grad_batches=args.accumulate_grad_batch,
                         check_val_every_n_epoch=args.write_every_epoch,
                         row_log_interval=args.row_log_interval,
                         log_save_interval=args.write_every_epoch,
                         nb_sanity_val_steps=1,
                         print_nan_grads=False)
    trainer.fit(VAE_metamorphic)

    logging.info(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
    logging.info('and going to http://localhost:6006 on your browser')

    # to restore model : https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.trainer.training_io.html

