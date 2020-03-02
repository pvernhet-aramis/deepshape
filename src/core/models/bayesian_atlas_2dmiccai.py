import os
import sys
import argparse
import datetime
import logging

### Visualization ###
import matplotlib
matplotlib.use('Agg')

### Core ###
import numpy as np
from torch.optim import Adam
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
from src.core.lightning_utils.callbacks import CustomModelCheckpoint

# ---------------------------------------------------------------------


class VariationalMetamorphicAtlas2dExecuter(pl.LightningModule):

    def __init__(self, hparams):
        super(VariationalMetamorphicAtlas2dExecuter, self).__init__()
        self.hparams = hparams
        self.check_hparams()

        # ---------- DATASETS
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

        # ----------  ATLAS TEMPLATE
        intensities_template, _ = dataset_train.compute_statistics()
        intensities_template = intensities_template.unsqueeze(0)       # (batch, channel, width, height)
        assert len(
            intensities_template.size()) == self.hparams.dimension + 2, "atlas size must be (batch, channel, width, height)"
        assert not torch.isnan(intensities_template).any(), "NaN detected"

        # ---------- MODEL
        self.model = MetamorphicAtlas2d(intensities_template, self.hparams.number_of_time_points,
                                        self.hparams.downsampling_data, self.hparams.downsampling_grid,
                                        self.hparams.latent_dimension__s, self.hparams.latent_dimension__a,
                                        self.hparams.kernel_width__s, self.hparams.kernel_width__a,
                                        initial_lambda_square__s=self.hparams.lambda_square__s,
                                        initial_lambda_square__a=self.hparams.lambda_square__a,
                                        noise_variance=self.hparams.noise_variance)

        # ---------- ADDITIONAL HOLDERS
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.ss_s_var = None
        self.ss_a_var = None
        self.attachment_loss = None
        self.last_device = None
        self.previous_atlas = intensities_template.clone().detach()    # on CPU

    def check_hparams(self):
        assert isinstance(self.hparams.num_workers, int) and self.hparams.num_workers >= 0, "num workers must be int"
        assert self.hparams.which_print in ["train", "test", "both"], "which print value not correct"

    def forward(self, x):
        return self.model.decode(x)

    def training_step(self, batch, batch_idx):
        """
        Variational Autoencoder step : KL divergence loss
        """
        self.last_device = batch.device.index
        batch_target_intensities = batch
        bts = batch_target_intensities.size(0)

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
        transformed_template = self.model(batch_latent__s, batch_latent__a)

        # ---------- LOSS AVERAGED BY PIXEL
        attachment_loss = self.mse(transformed_template, batch_target_intensities)
        kl_loss__s = torch.sum(
            (means__s.pow(2) + log_variances__s.exp()) / self.model.lambda_square__s - log_variances__s + np.log(
                self.model.lambda_square__s))
        kl_loss__a = torch.sum(
            (means__a.pow(2) + log_variances__a.exp()) / self.model.lambda_square__a - log_variances__a + np.log(
                self.model.lambda_square__a))

        total_loss = (attachment_loss / self.model.noise_variance + kl_loss__s + kl_loss__a) / bts

        # ---------- LOGS
        self.logger.experiment.add_scalars('attachment_loss', {'train': attachment_loss}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__s', {'train': kl_loss__s}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__a', {'train': kl_loss__a}, self.global_step)
        self.logger.experiment.add_scalars('total_loss', {'train': total_loss}, self.global_step)
        self.logger.experiment.add_scalars('ss_s_mean', {'train': ss_s_mean}, self.global_step)
        self.logger.experiment.add_scalars('ss_a_mean', {'train': ss_a_mean}, self.global_step)
        self.logger.experiment.add_scalars('ss_s_var', {'train': ss_s_var}, self.global_step)
        self.logger.experiment.add_scalars('ss_a_var', {'train': ss_a_var}, self.global_step)
        self.logger.experiment.add_scalar('lr', self.trainer.optimizers[0].param_groups[0]['lr'], self.global_step)

        # ---------- KEEP TRACKS FOR PARAMS CUSTOMIZED UPDATES
        self.ss_a_var = float(ss_a_var)
        self.ss_s_var = float(ss_s_var)
        self.attachment_loss = float(gpu_numpy_detach(attachment_loss) / self.model.noise_variance / bts)
        self.logger.experiment.add_scalars('monitored_training_loss', {'train': attachment_loss / bts}, self.global_step)

        return {'loss': total_loss}

    def on_after_backward(self):
        """
        Hyper-parameters update | batch_level
        """
        # -------- LOG ATLAS VARIATIONS
        atlas_delta = torch.sum(torch.abs(self.previous_atlas - self.model.template_intensities.cpu()))
        self.logger.experiment.add_scalar('atlas_total_L1_variations', atlas_delta, self.global_step)
        self.previous_atlas = self.model.template_intensities.clone().detach().cpu()

        # -------- UPDATE PARAMETERS IF NECESSARY
        if self.trainer.current_epoch >= self.hparams.update_from_epoch >= 1:
            self.model.noise_variance *= float(self.attachment_loss) / float(self.model.noise_dimension)
            self.model.lambda_square__a = float(self.ss_a_var)
            self.model.lambda_square__s = float(self.ss_s_var)

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
        transformed_template = self.model(batch_latent__s, batch_latent__a)

        # ---------- LOSS AVERAGED BY PIXEL
        attachment_loss = self.mse(transformed_template, batch_target_intensities)
        kl_loss__s = torch.sum(
            (means__s.pow(2) + log_variances__s.exp()) / self.model.lambda_square__s - log_variances__s + np.log(
                self.model.lambda_square__s))
        kl_loss__a = torch.sum(
            (means__a.pow(2) + log_variances__a.exp()) / self.model.lambda_square__a - log_variances__a + np.log(
                self.model.lambda_square__a))

        total_loss = (attachment_loss / self.model.noise_variance + kl_loss__s + kl_loss__a) / (bts * space_size)

        outputs = {
            'val_attachment_loss': attachment_loss,
            'val_kl_loss__s': kl_loss__s,
            'val_kl_loss__a': kl_loss__a,
            'val_total_loss': total_loss,
            'monitored_validation_loss': attachment_loss / bts
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
        monitored_validation_loss_mean = 0
        for output in outputs:
            val_attachment_loss_mean += output['val_attachment_loss']
            val_kl_loss__s_mean += output['val_kl_loss__s']
            val_kl_loss__a_mean += output['val_kl_loss__a']
            val_total_loss_mean += output['val_total_loss']
            monitored_validation_loss_mean += output['monitored_validation_loss']

        val_attachment_loss_mean /= len(outputs)
        val_kl_loss__s_mean /= len(outputs)
        val_kl_loss__a_mean /= len(outputs)
        val_total_loss_mean /= len(outputs)
        monitored_validation_loss_mean /= len(outputs)

        self.logger.experiment.add_scalars('attachment_loss', {'val': val_attachment_loss_mean}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__s', {'val': val_kl_loss__s_mean}, self.global_step)
        self.logger.experiment.add_scalars('kl_loss__a', {'val': val_kl_loss__a_mean}, self.global_step)
        self.logger.experiment.add_scalars('total_loss', {'val': val_total_loss_mean}, self.global_step)
        # self.logger.experiment.add_scalars('monitored_validation_loss', gpu_numpy_detach(monitored_validation_loss_mean), self.global_step)

        return {'val_loss': val_total_loss_mean, 'progress_bar': {'val_loss': val_total_loss_mean}}

    def configure_optimizers(self):
        base_opt = Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
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
            if self.hparams.which_print == 'train':
                self.save_viz(dataloader_name='train')
            elif self.hparams.which_print == 'test':
                self.save_viz(dataloader_name='test')
            elif self.hparams.which_print == 'both':
                self.save_viz(dataloader_name='train')
                self.save_viz(dataloader_name='test')
            else:
                raise AssertionError

    # def _save_model(self, checkpoint_file_path):
    #     """
    #     Checkpoints: (hparams | model eights)
    #     https://github.com/PyTorchLightning/PyTorch-Lightning/blob/master/pytorch_lightning/callbacks/model_checkpoint.py#L11-L181
    #     """
    #     os.makedirs(os.path.dirname(checkpoint_file_path), exist_ok=True)
    #     torch.save(self.model.state_dict(), os.path.join(self.hparams.snapshots_path,
    #                                                      'model__epoch_%d.pth' % self.current_epoch))
    #     # self.hparams | to be saved in csv format

    def save_viz(self, dataloader_name):
        """
        Saving nifti images
        """
        # Randomly select images
        assert dataloader_name in ['train', 'test']
        if dataloader_name == 'train':
            data_loader = self.train_dataloader()
            n = min(5, self.hparams.nb_train)
        else:
            data_loader = self.test_dataloader()[0]
            n = min(5, self.hparams.nb_test)
        intensities_to_write = []
        for batch_idx, intensities in enumerate(data_loader):
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
        self.model.write(intensities_to_write, os.path.join(self.hparams.snapshots_path,
                                                            '{}__epoch_{}'.format(dataloader_name, self.current_epoch)),
                         is_half=self.hparams.use_16bits)
        print('>> Save ', dataloader_name)


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
    parser.add_argument('--kernel_width__s', type=float, default=5, help='Kernel width s.')
    parser.add_argument('--kernel_width__a', type=float, default=2.5, help='Kernel width a.')
    parser.add_argument('--lambda_square__s', type=float, default=10. ** 2, help='Lambda square s.')
    parser.add_argument('--lambda_square__a', type=float, default=10. ** 2, help='Lambda square a.')
    parser.add_argument('--noise_variance', type=float, default=0.1 ** 2, help='Noise variance.')
    parser.add_argument('--downsampling_grid', type=int, default=2**1, choices=[1, 2, 4],
                        help='2**downsampling of grid.')
    parser.add_argument('--number_of_time_points', type=int, default=5, help='Integration time points.')
    # Training parameters
    parser.add_argument('--clipvar_min', type=float, default=float(-10*np.log(10)), help='10**min clip variance.')
    parser.add_argument('--clipvar_max', type=float, default=float(6*np.log(10)), help='10**max clip variance.')
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
    parser.add_argument("--optimizer", type=str, default='Adam', choices=['Adam'],
                        help="Adam optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam first order momentum decay")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam second order momentum decay")
    parser.add_argument('--step_lr', type=int, default=500, help='learning rate scheduler every epoch activation.')
    parser.add_argument('--step_decay', type=float, default=.75, help='learning rate scheduler decay value.')
    parser.add_argument('--update_from_epoch', type=int, default=-1, help='When to update lambdas.')
    # Checkpoints parameters
    parser.add_argument("--monitor_train", type=str, default='monitored_training_loss',
                        help="Monitored quantity for best training checkpoints.")
    parser.add_argument("--monitor_val", type=str, default='monitored_val_loss',
                        help="Monitored quantity for best validation checkpoints.")
    parser.add_argument("--save_top_k_train", type=int, default=0, help="How many top k training models saved.")
    parser.add_argument("--save_top_k_val", type=int, default=1, help="How many top k validation models saved.")
    parser.add_argument("--save_weights_only", action='store_true', help="Whether to save only weights or whole model.")
    parser.add_argument("--mode_train", type=str, default='min', choices=['min', 'max'], help="Op for training loss.")
    parser.add_argument("--mode_val", type=str, default='min', choices=['min', 'max'], help="Op for validation loss.")
    parser.add_argument("--period", type=int, default=1, help="Period at which to perform checkpoints.")
    # Storing data parameters
    parser.add_argument('--write_every_epoch', type=int, default=50, help='Number of iterations for checkpoints.')
    parser.add_argument('--row_log_interval', type=int, default=10, help='Log interval.')
    parser.add_argument('--which_print', type=str, default='both', choices=["train", "test", "both"],
                        help='From which dataset to print images.')
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
    os.makedirs(args.snapshots_path, exist_ok=True)
    print('\n>> Setting output directory to:\n', args.snapshots_path)

    # ==================================================================================================================
    # RUN TRAINING
    # ==================================================================================================================

    VAE_metamorphic = VariationalMetamorphicAtlas2dExecuter(args)

    custom_checkpoint_callback = CustomModelCheckpoint(filepath=os.path.join(args.snapshots_path, 'checkpoints'),
                                                       monitor_train=args.monitor_train,
                                                       monitor_val=args.monitor_val,
                                                       save_top_k_train=args.save_top_k_train,
                                                       save_top_k_val=args.save_top_k_val,
                                                       save_weights_only=args.save_weights_only,
                                                       mode_train=args.mode_train,
                                                       mode_val=args.mode_val,
                                                       period=args.period)
    custom_early_stopping = None

    trainer = pl.Trainer(gpus=([args.num_gpu] if args.cuda else None),
                         default_save_path=args.snapshots_path,
                         max_epochs=args.epochs,
                         early_stop_callback=custom_early_stopping,
                         checkpoint_callback=custom_checkpoint_callback,
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

