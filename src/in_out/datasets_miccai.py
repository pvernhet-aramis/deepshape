### Base ###
import fnmatch
import os
import math

### Core ###
import numpy as np
import torch
from tqdm import tqdm

### IMPORTS ###
from src.in_out.data_iclr import *
from torchvision import datasets, transforms
from scipy.ndimage import zoom
import nibabel as nib
import PIL.Image as pimg
from torch.utils.data import Dataset, DataLoader
from src.support.base_miccai import gpu_numpy_detach

def resize_32_32(image):
    a, b = image.shape
    return zoom(image, zoom=(32 * 1. / a, 32 * 1. / b))


# -----------------------------------------------------------
# 2D Datasets
# -----------------------------------------------------------


def load_mnist(number_of_images_train, number_of_images_test, digit=2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    number_of_images = number_of_images_train + number_of_images_test + 1
    mnist_data = datasets.MNIST('./', train=True, download=True)

    mnist_data = mnist_data.train_data[mnist_data.train_labels == digit]

    assert number_of_images <= mnist_data.shape[0], \
        'Too many required files. A maximum of %d are available' % mnist_data.shape[0]

    mnist_data__rdm = mnist_data[np.random.choice(mnist_data.shape[0], size=number_of_images, replace=None)]

    intensities = []
    for k, mnist_datum in enumerate(mnist_data__rdm):
        img = mnist_datum
        resized_img = resize_32_32(img)
        intensities.append(torch.from_numpy(resized_img).float())

    intensities = torch.stack(intensities)

    intensities_train = intensities[:number_of_images_train].unsqueeze(1)
    intensities_test = intensities[number_of_images_train:number_of_images_train + number_of_images_test].unsqueeze(1)
    intensities_template = intensities[-1].unsqueeze(0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = (intensities_train - intensities_mean) / intensities_std
    intensities_test = (intensities_test - intensities_mean) / intensities_std
    intensities_template = (intensities_template - intensities_mean) / intensities_std

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std


def load_eyes_black(data_path, number_of_images_train, number_of_images_test, random_seed=None):

    #path_to_train = os.path.normpath(
    #    os.path.join(os.path.dirname(__file__), '../../../../../../Data/MICCAI_dataset/eyes/data_final_4/train'))
    #path_to_test = os.path.normpath(
    #    os.path.join(os.path.dirname(__file__), '../../../../../../Data/MICCAI_dataset/eyes/data_final_4/test'))

    path_to_train = os.path.join(data_path, 'train')
    path_to_test = os.path.join(data_path, 'test')

    # Train
    intensities_train = []
    elts = sorted([_ for _ in os.listdir(path_to_train) if _[:3] == 'eye'], key=(lambda x: int(x.split('_')[-1][:-4])))
    for elt in elts:
        img = np.array(pimg.open(os.path.join(path_to_train, elt)))
        intensities_train.append(torch.from_numpy(img).float())
    intensities_train = torch.stack(intensities_train)

    assert number_of_images_train <= intensities_train.size(0), \
        'Too many required files. A maximum of %d are available' % intensities_train.size(0)
    intensities_train = intensities_train[:number_of_images_train]

    # Test
    intensities_test = []
    elts = sorted([_ for _ in os.listdir(path_to_test) if _[:3] == 'eye'], key=(lambda x: int(x.split('_')[-1][:-4])))
    for elt in elts:
        img = np.array(pimg.open(os.path.join(path_to_test, elt)))
        intensities_test.append(torch.from_numpy(img).float())
    intensities_test = torch.stack(intensities_test)

    assert number_of_images_test <= intensities_test.size(0), \
        'Too many required files. A maximum of %d are available' % intensities_test.size(0)
    intensities_test = intensities_test[:number_of_images_test]

    # Finalize
    intensities_template = torch.mean(intensities_train, dim=0)
    # intensities_template = torch.from_numpy(np.array(pimg.open(
    #     os.path.join(os.path.dirname(path_to_train), 'rectified_template.png')
    # ))).float()

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = ((intensities_train - intensities_mean) / intensities_std).unsqueeze(1)
    intensities_test = ((intensities_test - intensities_mean) / intensities_std).unsqueeze(1)
    intensities_template = ((intensities_template - intensities_mean) / intensities_std).unsqueeze(0)

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std


def load_brats(data_path, number_of_images_train, number_of_images_test, random_seed=None):

    path_to_train = os.path.join(data_path, 'train')
    path_to_test = os.path.join(data_path, 'test')

    # Train
    intensities_train = []

    elts = sorted([_ for _ in os.listdir(path_to_train) if _[-3:] == 'png'], key=(lambda x: int(x.split('_')[0])))
    for elt in elts:
        img = np.array(pimg.open(os.path.join(path_to_train, elt)))
        intensities_train.append(torch.from_numpy(img).float())
    intensities_train = torch.stack(intensities_train)

    assert number_of_images_train <= intensities_train.size(0), \
        'Too many required train files. A maximum of %d are available' % intensities_train.size(0).shape[0]
    intensities_train = intensities_train[:number_of_images_train]

    # Test
    intensities_test = []

    elts = sorted([_ for _ in os.listdir(path_to_test) if _[-3:] == 'png'], key=(lambda x: int(x.split('_')[0])))
    for elt in elts:
        img = np.array(pimg.open(os.path.join(path_to_test, elt)))
        intensities_test.append(torch.from_numpy(img).float())
    intensities_test = torch.stack(intensities_test)

    assert number_of_images_test <= intensities_test.size(0), \
        'Too many required test files. A maximum of %d are available' % intensities_test.size(0).shape[0]
    intensities_test = intensities_test[:number_of_images_test]

    # Finalize
    intensities_template = torch.mean(intensities_train, dim=0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = ((intensities_train - intensities_mean) / intensities_std).unsqueeze(1)
    intensities_test = ((intensities_test - intensities_mean) / intensities_std).unsqueeze(1)
    intensities_template = ((intensities_template - intensities_mean) / intensities_std).unsqueeze(0)

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std


def create_cross_sectional_brains_dataset__64(data_path, template_file, number_of_images_train, number_of_images_test,
                                              random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    path_to_data = data_path
    path_to_template = os.path.join(path_to_data, template_file)

    # CN.
    number_of_datum = number_of_images_train + number_of_images_test
    files = fnmatch.filter(os.listdir(path_to_data), 's*.nii')
    files = np.array(sorted(files))
    assert number_of_datum <= files.shape[0], \
        'Too many required CN brains. A maximum of %d are available' % files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

    intensities_cn = []
    for k, fl in enumerate(files__rdm):
        path_to_datum = os.path.join(path_to_data, fl)
        intensities_cn.append(torch.from_numpy(nib.load(path_to_datum).get_data()).float())
    intensities = torch.stack(intensities_cn).unsqueeze(1)

    intensities_train = intensities[:number_of_images_train]
    intensities_test = intensities[number_of_images_train:]
    intensities_template = torch.from_numpy(nib.load(path_to_template).get_data()).float().unsqueeze(0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = (intensities_train[:, :, :, :, 32] - intensities_mean) / intensities_std
    intensities_test = (intensities_test[:, :, :, :, 32] - intensities_mean) / intensities_std
    intensities_template = (intensities_template[:, :, :, 32] - intensities_mean) / intensities_std

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std


def create_cross_sectional_brains_dataset__128(data_path, template_file, number_of_datum_train, number_of_datum_test,
                                               random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    path_to_data = data_path
    path_to_template = os.path.join(path_to_data, template_file)

    number_of_datum_train_ = (number_of_datum_train // 3 + number_of_datum_train % 3,
                              number_of_datum_train // 3, number_of_datum_train // 3)
    number_of_datum_test_ = (number_of_datum_test // 3 + number_of_datum_test % 3,
                             number_of_datum_test // 3, number_of_datum_test // 3)

    print('>> TRAIN: %d CN ; %d AD ; %d MCI' %
          (number_of_datum_train_[0], number_of_datum_train_[1], number_of_datum_train_[2]))
    print('>> TEST : %d CN ; %d AD ; %d MCI' %
          (number_of_datum_test_[0], number_of_datum_test_[1], number_of_datum_test_[2]))

    # CN.
    number_of_datum = number_of_datum_train_[0] + number_of_datum_test_[0]
    if number_of_datum > 0:
        path_to_data_ = os.path.join(path_to_data, 'cn')
        files = fnmatch.filter(os.listdir(path_to_data_), 's*.npy')
        files = np.array(sorted(files))
        assert number_of_datum <= files.shape[0], \
            'Too many required CN brains. A maximum of %d are available' % files.shape[0]
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

        intensities_cn = []
        for k, fl in enumerate(files__rdm):
            path_to_datum = os.path.join(path_to_data_, fl)
            # intensities_cn.append(torch.from_numpy(np.transpose(np.load(path_to_datum)[::-1, ::-1]).copy()).float())
            intensities_cn.append(torch.from_numpy(np.load(path_to_datum)).float())
        intensities_cn = torch.stack(intensities_cn)

    # AD.
    number_of_datum = number_of_datum_train_[1] + number_of_datum_test_[1]
    if number_of_datum > 0:
        path_to_data_ = os.path.join(path_to_data, 'ad')
        files = fnmatch.filter(os.listdir(path_to_data_), 's*.npy')
        files = np.array(sorted(files))
        assert number_of_datum <= files.shape[0], \
            'Too many required AD brains. A maximum of %d are available' % files.shape[0]
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

        intensities_ad = []
        for k, fl in enumerate(files__rdm):
            path_to_datum = os.path.join(path_to_data_, fl)
            # intensities_ad.append(torch.from_numpy(np.transpose(np.load(path_to_datum)[::-1, ::-1]).copy()).float())
            intensities_ad.append(torch.from_numpy(np.load(path_to_datum)).float())
        intensities_ad = torch.stack(intensities_ad)

    # MCI.
    number_of_datum = number_of_datum_train_[2] + number_of_datum_test_[2]
    if number_of_datum > 0:
        path_to_data_ = os.path.join(path_to_data, 'mci')
        files = fnmatch.filter(os.listdir(path_to_data_), 's*.npy')
        files = np.array(sorted(files))
        assert number_of_datum <= files.shape[0], \
            'Too many required MCI brains. A maximum of %d are available' % files.shape[0]
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_datum, replace=None)]

        intensities_mci = []
        for k, fl in enumerate(files__rdm):
            path_to_datum = os.path.join(path_to_data_, fl)
            # intensities_mci.append(torch.from_numpy(np.transpose(np.load(path_to_datum)[::-1, ::-1]).copy()).float())
            intensities_mci.append(torch.from_numpy(np.load(path_to_datum)).float())
        intensities_mci = torch.stack(intensities_mci)

    intensities_train = torch.cat((intensities_cn[:number_of_datum_train_[0]],
                                   intensities_ad[:number_of_datum_train_[1]],
                                   intensities_mci[:number_of_datum_train_[2]]), dim=0).unsqueeze(1)
    intensities_test = torch.cat((intensities_cn[number_of_datum_train_[0]:],
                                  intensities_ad[number_of_datum_train_[1]:],
                                  intensities_mci[number_of_datum_train_[2]:]), dim=0).unsqueeze(1)
    intensities_template = torch.from_numpy(np.load(path_to_template)).float().unsqueeze(0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = (intensities_train[:, :, :, :, 60] - intensities_mean) / intensities_std
    intensities_test = (intensities_test[:, :, :, :, 60] - intensities_mean) / intensities_std
    intensities_template = (intensities_template[:, :, :, 60] - intensities_mean) / intensities_std

    # idx_u = torch.LongTensor([i for i in range(intensities_template.size(1) - 1, -1, -1)])
    # idx_v = torch.LongTensor([i for i in range(intensities_template.size(2) - 1, -1, -1)])
    # intensities_train = intensities_train.index_select(2, idx_u).index_select(3, idx_v).transpose(2, 3).contiguous()
    # intensities_test = intensities_test.index_select(2, idx_u).index_select(3, idx_v).transpose(2, 3).contiguous()
    # intensities_template = intensities_template.index_select(1, idx_u).index_select(2, idx_v).transpose(1, 2).contiguous()
    intensities_train = intensities_train.transpose(2, 3).contiguous()
    intensities_test = intensities_test.transpose(2, 3).contiguous()
    intensities_template = intensities_template.transpose(1, 2).contiguous()

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std


# -----------------------------------------------------------
# custom 3D DataLoader (avoids memory overload)
# -----------------------------------------------------------


class TrilinearInterpolation:
    """Trilinear interpolation for reduction if reduction size is not one"""

    def __init__(self, reduction):
        self.red = reduction

    def __call__(self, x):
        reduced_tensor = torch.nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0),
                                                         scale_factor=1. / self.red,
                                                         mode='trilinear', align_corners=False).squeeze(
            0) if self.red > 1 \
            else x.unsqueeze(0)
        return reduced_tensor


class StandardizedT13DDataset(Dataset):
    """(Sub) Dataset of BraTs 3D already gathered into folder.
    Specific to standardization of dataset using training data only.
    Requires online computations (bit more tricky).
    """

    def __init__(self, img_dir, nb_files, reduction=0, init_seed=123, check_endswith='pt', eps=1e-5):
        """
        Args:
            img_dir (string): Input directory - must contain all torch Tensors.
            nb_files (int): number of subset data to randomly select.
            data_file (string): File name of the train/test split file.
            init_seed (int): initialization seed for random data selection.
            check_endswith (string, optional): check for files extension.
        """
        assert len(check_endswith), "must check for valid files extension"
        self.img_dir = img_dir
        self.reduction = reduction
        self.base_transform = TrilinearInterpolation(self.reduction)
        self.normalization = None
        self.mean = None
        self.std = None
        self.eps = eps
        r = np.random.RandomState(init_seed)

        # Check path exists, and set nb_files to min if necessary
        if os.path.isdir(img_dir):
            candidates_tensors = [_ for _ in os.listdir(img_dir) if _.endswith(check_endswith)]
            nb_candidates = len(candidates_tensors)
            if nb_candidates < nb_files:
                print('>> Number of asked files {} exceeds number of available files {}'.format(nb_files, nb_candidates))
                print('>> Setting number of data to maximum available : {}'.format(nb_candidates))
                self.nb_files = nb_candidates
            else:
                print('>> Creating dataset with {} files (from {} available)'.format(nb_files, nb_candidates))
                self.nb_files = nb_files

            self.database = list(r.choice(candidates_tensors, size=self.nb_files, replace=False))
        else:
            raise Exception('The argument img_dir is not a valid directory.')

        # Computes mean and std statistics on dataset | to be used (or not)
        self.compute_statistics()

    def __len__(self):
        return self.nb_files

    def standardizer(self, image):
        """ default normalization"""
        return (image - self.mean) / self.std

    def set_normalization(self, normalization):
        self.normalization = normalization

    def __getitem__(self, idx):
        filename = self.database[idx]
        image_path = os.path.join(self.img_dir, filename)
        image = torch.load(image_path).float()
        transform = transforms.Compose([self.base_transform, self.normalization])
        sample = transform(image)  # (channel, width, height, depth) with channel = 1
        return sample

    def compute_statistics(self):
        """
        Computes statistics in an online fashion (using Welford’s method)
        """
        print('>> Computing online statistics for dataset ...')
        for elt in tqdm(range(self.nb_files)):
            sample = self.__getitem__(elt)
            image = sample.detach().clone()
            if elt == 0:
                current_mean = image
                current_var = torch.zeros_like(image)
            else:
                old_mean = current_mean.detach().clone()
                current_mean = old_mean + 1. / (1. + elt) * (image - old_mean)
                current_var = float(elt - 1) / float(elt) * current_var + 1. / (1. + elt) * (image - current_mean) * (image - old_mean)

        self.mean = current_mean.detach().clone().float()
        std = torch.sqrt(current_var).detach().clone().float()
        # ----------- Safety check for zero division
        std[np.where(std <= self.eps)] = self.eps
        self.std = std


class ZeroOneT13DDataset(Dataset):
    """(Sub) Dataset of BraTs 3D already gathered into folder.
    Rescaling of data to [0, 1] (uint8 + / 255)
    """

    def __init__(self, img_dir, nb_files, reduction=0, init_seed=123, check_endswith='pt', eps=1e-5):
        """
        Args:
            img_dir (string): Input directory - must contain all torch Tensors.
            nb_files (int): number of subset data to randomly select.
            data_file (string): File name of the train/test split file.
            init_seed (int): initialization seed for random data selection.
            check_endswith (string, optional): check for files extension.
        """
        assert len(check_endswith), "must check for valid files extension"
        self.img_dir = img_dir
        self.reduction = reduction
        self.base_transform = TrilinearInterpolation(self.reduction)
        self.transform = transforms.Compose([
            self.base_transform,
            transforms.Lambda(lambda x: (x.div(255).float()))     # .type(torch.uint8)/255
        ])
        self.eps = eps
        r = np.random.RandomState(init_seed)

        # Check path exists, and set nb_files to min if necessary
        if os.path.isdir(img_dir):
            candidates_tensors = [_ for _ in os.listdir(img_dir) if _.endswith(check_endswith)]
            nb_candidates = len(candidates_tensors)
            if nb_candidates < nb_files:
                print('>> Number of asked files {} exceeds number of available files {}'.format(nb_files, nb_candidates))
                print('>> Setting number of data to maximum available : {}'.format(nb_candidates))
                self.nb_files = nb_candidates
            else:
                print('>> Creating dataset with {} files (from {} available)'.format(nb_files, nb_candidates))
                self.nb_files = nb_files

            self.database = list(r.choice(candidates_tensors, size=self.nb_files, replace=False))
        else:
            raise Exception('The argument img_dir is not a valid directory.')

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = self.database[idx]
        image_path = os.path.join(self.img_dir, filename)
        image = torch.load(image_path).float()
        sample = self.transform(image)      # (channel, width, height, depth) with channel = 1
        return sample

    def compute_statistics(self):
        """
        Computes statistics in an online fashion (using Welford’s method)
        """
        print('>> Computing online statistics for dataset ...')
        for elt in tqdm(range(self.nb_files)):
            sample = self.__getitem__(elt)
            image = sample.detach().clone()
            if elt == 0:
                current_mean = image
                current_var = torch.zeros_like(image)
            else:
                old_mean = current_mean.detach().clone()
                current_mean = old_mean + 1. / (1. + elt) * (image - old_mean)
                current_var = float(elt - 1) / float(elt) * current_var + 1. / (1. + elt) * (image - current_mean) * (image - old_mean)

        mean = current_mean.detach().clone().float()
        std = torch.sqrt(current_var).detach().clone().float()
        # ----------- Safety check for zero division
        std[np.where(std <= self.eps)] = self.eps
        return mean, std

