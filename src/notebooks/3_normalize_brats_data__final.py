import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import fnmatch
from torch.utils.data import TensorDataset, DataLoader
import itertools
import math
from sklearn.decomposition import PCA

from scipy.ndimage import gaussian_filter
import PIL.Image as pimg
import nibabel as nib
from multiprocessing import Pool
import itk


args = []

path_to_dataset__in = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../2_datasets/1_brats_2019'))
path_to_dataset__out = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../2_datasets/2_t1ce_normalized'))
grades = ['HGG', 'LGG']
img_type = 't1ce'
path_to_file__ref = '3_colin27_t1_tal_lin_brain.nii'

if not os.path.isdir(path_to_dataset__out):
    os.mkdir(path_to_dataset__out)

### TRAIN ###
t = ('1_training', 'train')
path_to_t = os.path.join(path_to_dataset__out, t[1])
if not os.path.isdir(path_to_t):
    os.mkdir(path_to_t)
for grade in grades:
    path_to_subjects = os.path.join(path_to_dataset__in, t[0], grade)
    subject_ids = [elt for elt in os.listdir(path_to_subjects) if elt[:5] == 'BraTS']
    for subject_id in subject_ids:
        path_to_file__in = os.path.join(path_to_dataset__in, t[0], grade, subject_id, subject_id + '_%s.nii.gz' % img_type)
        path_to_file__out = os.path.join(path_to_dataset__out, t[1], grade.lower() + '_' + subject_id[8:] + '_' + img_type)
        args.append((subject_id, path_to_file__in, path_to_file__out + '.nii.gz'))

### TEST ###
t = ('2_validation', 'test')
path_to_t = os.path.join(path_to_dataset__out, t[1])
if not os.path.isdir(path_to_t):
    os.mkdir(path_to_t)
path_to_subjects = os.path.join(path_to_dataset__in, t[0])
subject_ids = [elt for elt in os.listdir(path_to_subjects) if elt[:5] == 'BraTS']
for subject_id in subject_ids:
    path_to_file__in = os.path.join(path_to_dataset__in, t[0], subject_id, subject_id + '_%s.nii.gz' % img_type)
    path_to_file__out = os.path.join(path_to_dataset__out, t[1], subject_id[8:] + '_' + img_type)
    args.append((subject_id, path_to_file__in, path_to_file__out + '.nii.gz'))

### RUN ###
PixelType = itk.F
ImageType = itk.Image[PixelType, 3]

img_ref = itk.imread(path_to_file__ref, PixelType)
# img_ref = itk.RescaleIntensityImageFilter(img_ref, ttype=(ImageType, ImageType))
# img_ref = itk.cast_image_filter(img_ref)

def launch(args):
    subject_id, path_in, path_out = args
    print(subject_id)
    img_in = itk.imread(path_in, PixelType)
#     img_in = itk.RescaleIntensityImageFilter(img_in, ttype=(ImageType, ImageType))
    img_out = itk.HistogramMatchingImageFilter(img_in, img_ref)
    img_out = itk.RescaleIntensityImageFilter(img_out)
    itk.imwrite(img_out, path_out)

# for arg in args:
#     launch(arg)

with Pool(os.cpu_count()) as pool:
    pool.map(launch, args)
