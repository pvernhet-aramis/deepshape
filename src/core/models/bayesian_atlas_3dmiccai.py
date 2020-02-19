import os
import sys
import argparse
import datetime
from copy import deepcopy
import logging
from collections import OrderedDict
import itertools
from functools import reduce
from operator import mul

### Visualization ###
import matplotlib
matplotlib.use('Agg')

### Core ###
import numpy as np
from torch.optim import Adam
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Setting paths to directory roots | >> deepshape
parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent)
os.chdir(parent)
print('Setting root path to : {}'.format(parent))

### IMPORTS ###
from src.in_out.datasets_miccai import ZeroOneT13DDataset
from src.support.nets_miccai_3d import MetamorphicAtlas
from src.support.base_miccai import *


# ---------------------------------------------------------------------


class VariationalMetamorphicAtlasExecuter(pl.LightningModule):
    """
    See doc https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
    """

    def __init__(self, hparams, model, affine):
        super(VariationalMetamorphicAtlasExecuter, self).__init__()
        self.hparams = hparams
        self.check_hparams()

        # nn.Module parameters
        self.model = model
        self.ss_s_var = None
        self.ss_a_var = None
        self.affine = affine

        self.epoch = 0

    def check_hparams(self):
        assert isinstance(self.hparams.num_workers, int) and self.hparams.num_workers >= 1, "num workers must be int"

    def forward(self, x):
        return self.model.decode(x)

    def training_step(self, batch, batch_idx):
        """
        Variational Autoencoder step : KL divergence loss
        """
        batch_target_intensities = batch
        bts = batch_target_intensities.size(0)
        space_size = reduce(mul, batch_target_intensities.size()[2:])
        per_voxel_per_batch = float(bts * space_size)

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
        attachment_loss = torch.sum((transformed_template - batch_target_intensities) ** 2) / self.model.noise_variance
        kl_loss__s = torch.sum(
            (means__s.pow(2) + log_variances__s.exp()) / self.model.lambda_square__s - log_variances__s + np.log(
                self.model.lambda_square__s))
        kl_loss__a = torch.sum(
            (means__a.pow(2) + log_variances__a.exp()) / self.model.lambda_square__a - log_variances__a + np.log(
                self.model.lambda_square__a))

        total_loss = (attachment_loss + kl_loss__s + kl_loss__a) / per_voxel_per_batch

        outputs = {'train_attachment_loss': attachment_loss,
            'train_kl_loss__s': kl_loss__s,
            'train_kl_loss__a': kl_loss__a,
            'train_total_loss': total_loss,
            'train_ss_s_mean': ss_s_mean,
            'train_ss_a_mean': ss_a_mean,
            'train_ss_s_var': ss_s_var,
            'train_ss_a_var': ss_a_var,
        }

        tensorboard_logs = {
            'train/attachment_loss': attachment_loss,
            'train/kl_loss__s': kl_loss__s,
            'train/kl_loss__a': kl_loss__a,
            'train/total_loss': total_loss,
            'train/ss_s_mean': ss_s_mean,
            'train/ss_a_mean': ss_a_mean,
            'train/ss_s_var': ss_s_var,
            'train/ss_a_var': ss_a_var,
        }

        return {'outputs': outputs, 'log': tensorboard_logs}

    def on_train_end(self):
        """
        Called on end of training step
        """

        if self.epoch % self.update_per_batch:
            # Update parameters
            self.model.noise_variance *= None
            self.model.lambda_square__a = None
            self.model.lambda_square__s = None

    def on_epoch_end(self):
        """
        Called on end of training epoch
        """

        if self.epoch == 0 or self.epoch % self.hparams.write_every_epoch:
            self.save_viz()
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        """
        Variational Autoencoder step : KL divergence
        """

        batch_target_intensities = batch
        bts = batch_target_intensities.size(0)
        space_size = reduce(mul, batch_target_intensities.size()[2:])
        per_voxel_per_batch = float(bts * space_size)

        # ---------- ENCODE, SAMPLE AND DECODE
        means__s, log_variances__s, means__a, log_variances__a = self.model.encode(batch_target_intensities)
        log_variances__s = torch.clamp(log_variances__s, self.hparams.clipvar_min, self.hparams.clipvar_max)
        log_variances__a = torch.clamp(log_variances__a, self.hparams.clipvar_min, self.hparams.clipvar_max)
        stds__s, stds__a = torch.exp(0.5 * log_variances__s), torch.exp(0.5 * log_variances__a)

        batch_latent__s = means__s + torch.zeros_like(means__s).normal_() * stds__s
        batch_latent__a = means__a + torch.zeros_like(means__a).normal_() * stds__a
        transformed_template = model(batch_latent__s, batch_latent__a)

        # ---------- LOSS AVERAGED BY VOXEL
        attachment_loss = torch.sum((transformed_template - batch_target_intensities) ** 2) / self.model.noise_variance
        kl_loss__s = torch.sum(
            (means__s.pow(2) + log_variances__s.exp()) / self.model.lambda_square__s - log_variances__s + np.log(
                self.model.lambda_square__s))
        kl_loss__a = torch.sum(
            (means__a.pow(2) + log_variances__a.exp()) / self.model.lambda_square__a - log_variances__a + np.log(
                self.model.lambda_square__a))

        total_loss = (attachment_loss + kl_loss__s + kl_loss__a) / per_voxel_per_batch

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

        tensorboard_logs = {
            'val/attachment_loss': val_attachment_loss_mean.item(),
            'val/kl_loss__s': val_kl_loss__s_mean.item(),
            'val/kl_loss__a': val_kl_loss__a_mean.item(),
            'val/total_loss': val_total_loss_mean.item(),
        }
        results = {'log': tensorboard_logs}
        return results

    def configure_optimizers(self):
        optimizers = [Adam(self.model.parameters(), self.hparams.lr)]
        schedulers = [ReduceLROnPlateau(optimizers[0], factor=self.hparams.lr_decay, patience=self.hparams.lr_patience,
                                        min_lr=self.hparams.min_lr, verbose=True, threshold_mode='abs')]
        return optimizers, schedulers

    @pl.data_loader
    def train_dataloader(self):

        dataset_train = ZeroOneT13DDataset(os.path.join(self.hparams.data_tensor_path, 'train'), self.hparams.nb_train,
                                           reduction=self.hparams.reduction,  init_seed=self.hparams.seed,
                                           check_endswith='pt')
        train_loader = data_utils.DataLoader(dataset_train, batch_size=self.hparams.batch_size, shuffle=True,
                                             num_workers=self.hparams.num_workers)
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        dataset_val = ZeroOneT13DDataset(os.path.join(self.hparams.data_tensor_path, 'test'), self.hparams.nb_test,
                                          reduction=self.hparams.reduction, init_seed=self.hparams.seed,
                                          check_endswith='pt')
        val_loader = data_utils.DataLoader(dataset_val, batch_size=self.hparams.batch_size, shuffle=True,
                                            num_workers=self.hparams.num_workers)
        return val_loader

    @pl.data_loader
    def test_dataloader(self):
        dataset_test = ZeroOneT13DDataset(os.path.join(self.hparams.data_tensor_path, 'test'), self.hparams.nb_test,
                                          reduction=self.hparams.reduction, init_seed=self.hparams.seed,
                                          check_endswith='pt')
        test_loader = data_utils.DataLoader(dataset_test, batch_size=self.hparams.batch_size, shuffle=True,
                                            num_workers=self.hparams.num_workers)
        return test_loader

    def save_model(self):
        """
        Save model (akin to checkpoints)
        """
        torch.save(self.model.state_dict(), os.path.join(self.hparams.snapshots_path, 'model__epoch_%d.pth' % self.epoch))

    def save_viz(self):
        """
        Saving nifti images
        """
        # Randomly select images
        n = min(5, self.hparams.nb_test)
        intensities_to_write = []
        for batch_idx, intensities in enumerate(self.test_dataloader):
            if n <= 0:
                break
            bts = intensities.size(0)
            nb_selected = min(bts, n)
            intensities_to_write.append(intensities[:nb_selected])
            n = n - nb_selected
        intensities_to_write = torch.cat(intensities_to_write)
        self.model.write(intensities_to_write, os.path.join(self.hparams.snapshots_path, 'train__epoch_%d' % self.epoch),
                         affine=self.affine)
        print('>> Saving done')


if __name__ == '__main__':

    # ==================================================================================================================
    # GLOBAL VARIABLES
    # ==================================================================================================================

    parser = argparse.ArgumentParser(description='Bayesian 3D Atlas MICCAI 2020 | LIGHTNING VERSION.')
    # action parameters
    parser.add_argument('--data_dir', type=str, default='Data/MICCAI_dataset',
                        help='Data directory root.')
    parser.add_argument('--cuda', action='store_true', help='Whether CUDA is available on GPUs.')
    parser.add_argument('--num_gpu', type=int, default=0, help='Which GPU to run on.')
    parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use if cuda not available')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    # Dataset parameters
    parser.add_argument('--dimension', type=int, default=3, choices=[3], help='Dataset dimension.')
    parser.add_argument('--downsampling_data', type=int, default=2**1, choices=[1, 2, 4],
                        help='2**downsampling of initial data.')
    # Model parameters
    parser.add_argument('--latent_dimension__s', type=int, default=10, help='Latent dimension of s.')
    parser.add_argument('--latent_dimension__a', type=int, default=5, help='Latent dimension of a.')
    parser.add_argument('--kernel_width__s', type=int, default=5, help='Kernel width s.')
    parser.add_argument('--kernel_width__a', type=int, default=2.5, help='Kernel width a.')
    parser.add_argument('--lambda_square__s', type=float, default=10 ** 2, help='Lambda square s.')
    parser.add_argument('--lambda_square__a', type=float, default=10 ** 2, help='Lambda square a.')
    parser.add_argument('--noise_variance', type=float, default=0.1 ** 2, help='Noise variance.')
    parser.add_argument('--downsampling_grid', type=int, default=2**1, choices=[1, 2, 4],
                        help='2**downsampling of grid.')
    parser.add_argument('--number_of_time_points', type=int, default=5, help='Integration time points.')
    # Training parameters
    parser.add_argument('--clipvar_min', type=float, default=-5, help='10**min clip variance.')
    parser.add_argument('--clipvar_max', type=float, default=2, help='10**max clip variance.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to perform.')
    parser.add_argument('--nb_train', type=int, default=8, help='Number of training data.')
    parser.add_argument('--nb_test', type=int, default=1, help='Number of testing data.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size when processing data.')
    parser.add_argument('--accumulated_batch', type=int, default=4, help='Number of accumulated batch for grad step.')
    parser.add_argument('--use_16bits', action='store_true', help='Whether to use 16-bits mixed precision.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloaders.')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--lr_ratio', type=float, default=1, help='learning rate ratio.')
    parser.add_argument('--lr_decay', type=float, default=.5, help='learning rate decay.')
    parser.add_argument('--lr_patience', type=int, default=5, help='learning rate patience.')
    parser.add_argument('--min_lr', type=float, default=float(5e-5), help='minimal learning rate.')
    parser.add_argument('--early_patience', type=int, default=5, help='learning rate patience for early stopping.')
    parser.add_argument('--early_min', type=int, default=200, help='minimum epochs before early stopping.')
    # Storing data parameters
    parser.add_argument('--val_check_interval', type=int, default=2, help='Number of epoch iterations between eval.')
    parser.add_argument('--write_every_epoch', type=int, default=50, help='Number of iterations for checkpoints.')
    parser.add_argument('--track_norms', type=int, default=-1, help='Track gradients norms (default: None).')

    args = parser.parse_args()

    HOME_PATH = '/network/lustre/dtlake01/aramis/users/paul.vernhet'

    args.experiment_prefix = '3D_rdm_slice_normalization_{}_reduction'.format(args.downsampling_data)
    # data_nifti_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/2_datasets/2_t1ce_normalized')
    data_tensor_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/2_t1ce_normalized/0_reduction')
    args.output_dir = os.path.join(HOME_PATH, '3dBraTs', args.experiment_prefix)
    args.batch_size = min(args.batch_size, args.nb_train)
    args.accumulated_gradient_steps = args.accumulated_batch // args.batch_size
    np_affine = np.load(file=os.path.join(data_tensor_path, 'train', 'affine.npy'))

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
    args.snapshots_path = os.path.join(args.output_dir, '{}'.format(args.model_signature))
    if not os.path.exists(args.snapshots_path):
        os.makedirs(args.snapshots_path)
    print('\n>> Setting output directory to:\n', args.snapshots_path)

    # with open(os.path.join(args.snapshots_path, 'args.json'), 'w') as f:
    #     args_wo_device = deepcopy(args.__dict__)
    #     args_wo_device.pop('device', None)
    #     json.dump(args_wo_device, f, indent=4, sort_keys=True)

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================

    # INITIALIZE TEMPLATE TO MEAN OF TRAINING DATA ------------------------------
    dataset_train = ZeroOneT13DDataset(os.path.join(args.data_tensor_path, 'train'), args.nb_train,
                                       reduction=args.reduction, init_seed=args.seed,
                                       check_endswith='pt')
    intensities_template, _ = dataset_train.compute_statistics()
    intensities_template = intensities_template.unsqueeze(0)
    print('>> Templated initialized successfully\n')

    # ==================================================================================================================
    # BUILD MODEL
    # ==================================================================================================================

    model = MetamorphicAtlas(
        intensities_template, args.nb_train, args.downsampling_data, args.downsampling_grid,
        args.latent_dimension__s, args.latent_dimension__a,
        args.kernel_width__s, args.kernel_width__a,
        initial_lambda_square__s=args.lambda_square__s, initial_lambda_square__a=args.lambda_square__a).to(DEVICE)

    # ==================================================================================================================
    # RUN TRAINING
    # ==================================================================================================================

    VAE_metamorphic = VariationalMetamorphicAtlasExecuter(args, model, np_affine)

    custom_early_stop_callback = EarlyStopping(
        monitor='val_total_loss',
        min_delta=0.00,
        patience=args.early_patience,
        verbose=False,
        mode='min'
    ) if args.min_early < args.epochs else None

    trainer = pl.Trainer(gpus=[args.num_gpu],
                         default_save_path=args.snapshots_path,
                         min_epochs=args.min_early,
                         max_epochs=args.epochs,
                         early_stop_callback=custom_early_stop_callback,
                         use_amp=args.use_16bits,
                         amp_level='O2',
                         track_grad_norm=args.track_norms,
                         val_check_interval=int(args.nb_train/args.batch_size * args.val_check_interval),
                         accumulate_grad_batches=args.accumulated_gradient_steps)
    trainer.fit(VAE_metamorphic)

    logging.info(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
    logging.info('and going to http://localhost:6006 on your browser')

    # to restaure model : https://pytorch-lightning.readthedocs.io/en/latest/pytorch_lightning.trainer.training_io.html


