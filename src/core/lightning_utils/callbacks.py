import os
import shutil
import logging as log
import warnings

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class CustomModelCheckpoint(pl.callbacks.pt_callbacks.Callback):
    r"""
    >> Custom Lightning model checkpoint <<
    Saves the model after every epoch, potentially on both train and val values.
    Args:
        filepath: path to save the model file.
            Can contain named formatting options to be auto-filled.
            Example::
                # save epoch and val_loss in name
                ModelCheckpoint(filepath='{epoch:02d}-{val_loss:.2f}.hdf5')
                # saves file like: /path/epoch_2-val_loss_0.2.hdf5
        monitor (str): quantity to monitor.
        verbose (bool): verbosity mode, False or True.
        save_top_k (int): if `save_top_k == k`,
            the best k models according to
            the quantity monitored will be saved.
            if ``save_top_k == 0``, no models are saved.
            if ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if ``save_top_k >= 2`` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode (str): one of {auto, min, max}.
            If ``save_top_k != 0``, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only (bool): if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period (int): Interval (number of epochs) between checkpoints.
    Example::
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        # saves checkpoints to my_path whenever 'val_loss' has a new min
        checkpoint_callback = ModelCheckpoint(filepath='my_path')
        Trainer(checkpoint_callback=checkpoint_callback)
    """

    def __init__(self, filepath, monitor_train: str = 'train_loss', monitor_val: str = 'val_loss',
                 verbose: bool = False,
                 save_top_k_train: int = 1, save_top_k_val: int = 1, save_weights_only: bool = False,
                 mode_train: str = 'min', mode_val: str = 'min', period: int = 1, prefix: str = ''):
        super().__init__()

        assert mode_train in ['min', 'max'], "monitoring mode can only be min or max"
        assert mode_val in ['min', 'max'], "monitoring mode can only be min or max"

        if (save_top_k_train or save_top_k_val) and os.path.isdir(filepath) and len(os.listdir(filepath)) > 0:
            warnings.warn(
                f"Checkpoint directory {filepath} exists and is not empty with save_top_k_train of _val != 0."
                "All files in this directory will be deleted when a checkpoint is saved!"
            )

        self.verbose = verbose
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_check = 0
        os.makedirs(filepath, exist_ok=True)
        self.prefix = {'train': '_train',
                       'val': '_val'
                       }
        self.monitor_op = {'train': monitor_train,
                           'val': monitor_val
                           }
        self.save_top_k = {'train': save_top_k_train,
                           'val': save_top_k_val
                           }
        self.best_k_models = {'train': {},
                              'val': {}
                              }
        self.kth_best_model = {'train': '',
                               'val': ''
                               }
        self.best = {'train': 0,
                     'val': 0
                     }
        self.kth_value = {'train': 0,
                          'val': 0
                          }
        self.mode = {'train': '',
                     'val': ''
                     }
        self.save_function = None

        mode_dict = {
            'min': (np.less, np.Inf, 'min'),
            'max': (np.greater, -np.Inf, 'max')
        }

        self.monitor_op['train'], self.kth_value['train'], self.mode['train'] = mode_dict[mode_train]
        self.monitor_op['val'], self.kth_value['val'], self.mode['val'] = mode_dict[mode_val]
        print('CHECKPOINT CALLBACK INITIALIZED')

    def _del_model(self, filepath):
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    def _save_model(self, filepath):
        # make paths
        print('CALLING SAVE FUNCTION')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the model
        if self.save_function is not None:
            self.save_function(filepath)
        else:
            raise ValueError(".save_function() not set")

    def check_monitor_top_k(self, current, which_loop):
        assert which_loop in ['train', 'val']
        if len(self.best_k_models[which_loop]) < self.save_top_k[which_loop]:
            return True
        return self.monitor_op[which_loop](current, self.best_k_models[which_loop][self.kth_best_model[which_loop]])

    def on_custom_checkpoint_end(self, trainer, pl_module, which_loop):
        """
        Main custom checkpoint( valid for train and validation ends
        """
        assert which_loop in ['train', 'val']

        # TODO: Check epochs_since_last_check is not updated twice
        logs = trainer.callback_metrics
        epoch = trainer.current_epoch
        self.epochs_since_last_check += (1 if which_loop == 'train' else 0)
        print('>> On custom checkpoint end | ', which_loop)
        print(self.monitor[which_loop])
        print(logs.get(self.monitor[which_loop]))
        print('**')

        if self.save_top_k[which_loop] == 0:
            # no models are saved
            return
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = f'{self.filepath}/{self.prefix[which_loop]}_ckpt_epoch_{epoch}.ckpt'
            version_cnt = 0
            while os.path.isfile(filepath):
                # this epoch called before
                filepath = f'{self.filepath}/{self.prefix[which_loop]}_ckpt_epoch_{epoch}_v{version_cnt}.ckpt'
                version_cnt += 1

            if self.save_top_k[which_loop] != -1:
                current = logs.get(self.monitor[which_loop])

                if current is None:
                    warnings.warn(
                        f'Can save best model only with {self.monitor[which_loop]} available,'
                        ' skipping.', RuntimeWarning)
                else:
                    if self.check_monitor_top_k(current, which_loop):
                        self._do_check_save(filepath, current, epoch, which_loop)
                    else:
                        if self.verbose > 0:
                            log.info(
                                f'\nEpoch {epoch:05d}: {self.monitor[which_loop]}'
                                f' was not in top {self.save_top_k[which_loop]}')

            else:
                if self.verbose > 0:
                    log.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')
                self._save_model(filepath)

    def on_train_end(self, trainer, pl_module):
        """
        Calling update on pl.lightning_module on_train_end step
        """
        print('CALLBACK CALL (train)')
        self.on_custom_checkpoint_end(trainer, pl_module, which_loop='train')

    def on_validation_end(self, trainer, pl_module):
        """
        Calling update on pl.lightning_module on_validation_end step
        """
        print('CALLBACK CALL (validation)')
        self.on_custom_checkpoint_end(trainer, pl_module, which_loop='val')

    def _do_check_save(self, filepath, current, epoch, which_loop):
        # remove kth
        if len(self.best_k_models[which_loop]) == self.save_top_k[which_loop]:
            delpath = self.kth_best_model[which_loop]
            self.best_k_models[which_loop].pop(self.kth_best_model[which_loop])
            self._del_model(delpath)

        self.best_k_models[which_loop][filepath] = current
        if len(self.best_k_models[which_loop]) == self.save_top_k[which_loop]:
            # monitor dict has reached k elements
            _op = max if self.mode[which_loop] == 'min' else min
            self.kth_best_model[which_loop] = _op(self.best_k_models[which_loop],
                                                  key=self.best_k_models[which_loop].get)
            self.kth_value[which_loop] = self.best_k_models[which_loop][self.kth_best_model[which_loop]]

        _op = min if self.mode[which_loop] == 'min' else max
        self.best[which_loop] = _op(self.best_k_models[which_loop].values())

        if self.verbose > 0:
            log.info(
                f'\nEpoch {epoch:05d}: {self.monitor[which_loop]} reached'
                f' {current:0.5f} (best {self.best[which_loop]:0.5f}), saving model to'
                f' {filepath} as top {self.save_top_k[which_loop]}')
        self._save_model(filepath)

