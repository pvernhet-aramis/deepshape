### Base ###
import fnmatch
import os

### Core ###
import numpy as np
import torch

### IMPORTS ###
from in_out.data_iccv import *


def extract_subject_and_visit_ids(filename):
    out = []
    if filename[50:52] == '__':
        out.append(int(filename[49]))
        if filename[56:58] == '__':
            out.append(int(filename[55]))
        else:
            out.append(int(filename[55:57]))
    else:
        out.append(int(filename[49:51]))
        if filename[57:59] == '__':
            out.append(int(filename[56]))
        else:
            out.append(int(filename[56:58]))
    return out


def create_cross_sectional_starmen_dataset(path_to_starmen, number_of_starmen_train, number_of_starmen_test,
                                           splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_starmen = number_of_starmen_train + number_of_starmen_test

    starmen_files = fnmatch.filter(os.listdir(path_to_starmen), 'SimulatedData__Reconstruction__starman__subject_*')
    starmen_files = np.array(sorted(starmen_files, key=extract_subject_and_visit_ids))

    assert number_of_starmen <= starmen_files.shape[0], 'Too many required starmen. A maximum of %d are available' % \
                                                        starmen_files.shape[0]
    starmen_files__rdm = starmen_files[np.random.choice(starmen_files.shape[0], size=number_of_starmen, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, starman_file in enumerate(starmen_files__rdm):
        path_to_starman = os.path.join(path_to_starmen, starman_file)

        p, c = read_vtk_file(path_to_starman, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    splats = (splats - torch.mean(splats)) / torch.std(splats)

    return (points[:number_of_starmen_train],
            connectivities[:number_of_starmen_train],
            splats[:number_of_starmen_train],
            points[number_of_starmen_train:],
            connectivities[number_of_starmen_train:],
            splats[number_of_starmen_train:])


def create_cross_sectional_ellipsoids_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                              splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'ellipsoid*')
    files = np.array(sorted(files))

    assert number_of_meshes <= files.shape[0], 'Too many required ellipsoids. A maximum of %d are available' % \
                                               files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, starman_file in enumerate(files__rdm):
        path_to_starman = os.path.join(path_to_meshes, starman_file)

        p, c = read_vtk_file(path_to_starman, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])


def create_cross_sectional_hippocampi_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                              splatting_grid, dimension, gkernel, gamma, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'sub-ADNI*')
    files = np.array(sorted(files))

    assert number_of_meshes <= files.shape[0], 'Too many required hippocampi. A maximum of %d are available' % \
                                               files.shape[0]

    if number_of_meshes < len(files):
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]
    else:
        files__rdm = files

    points = []
    connectivities = []
    centers = []
    normals = []
    norms = []
    splats = []
    for k, fl in enumerate(files__rdm):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        ctr, nrl = compute_centers_and_normals(p, c)
        # s = convolve(splatting_grid.view(-1, dimension), ctr, nrl, kernel_width).view(splatting_grid.size())
        s = gkernel(gamma, splatting_grid.view(-1, dimension), ctr, nrl).view(splatting_grid.size())
        points.append(p)
        connectivities.append(c)
        centers.append(ctr)
        normals.append(nrl)
        norms.append(torch.sum(nrl * gkernel(gamma, ctr, ctr, nrl)))
        splats.append(s)

    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            centers[:number_of_meshes_train],
            normals[:number_of_meshes_train],
            norms[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            centers[number_of_meshes_train:],
            normals[number_of_meshes_train:],
            norms[number_of_meshes_train:],
            splats[number_of_meshes_train:])


def create_cross_sectional_circles_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                           splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'circle*')
    files = np.array(sorted(files))

    assert number_of_meshes <= files.shape[0], 'Too many required circles. A maximum of %d are available' % \
                                               files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, fl in enumerate(files__rdm):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])


def create_cross_sectional_leaves_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                          splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'leaf*')
    files = np.array(sorted(files))

    assert number_of_meshes <= files.shape[0], 'Too many required leaves. A maximum of %d are available' % \
                                               files.shape[0]
    files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]

    points = []
    connectivities = []
    splats = []
    for k, fl in enumerate(files__rdm):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])


def create_cross_sectional_squares_dataset(path_to_meshes, number_of_meshes_train, number_of_meshes_test,
                                           splatting_grid, kernel_width, dimension, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    number_of_meshes = number_of_meshes_train + number_of_meshes_test

    files = fnmatch.filter(os.listdir(path_to_meshes), 'square*')
    files = np.array(sorted(files, key=(lambda x: [int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])])))

    assert number_of_meshes <= files.shape[0], 'Too many required squares. A maximum of %d are available' % \
                                               files.shape[0]

    if number_of_meshes < len(files):
        files__rdm = files[np.random.choice(files.shape[0], size=number_of_meshes, replace=None)]
    else:
        files__rdm = files

    points = []
    connectivities = []
    splats = []
    for k, fl in enumerate(files__rdm):
        path_to_mesh = os.path.join(path_to_meshes, fl)

        p, c = read_vtk_file(path_to_mesh, dimension=dimension, extract_connectivity=True)
        s = splat_current_on_grid(p, c, splatting_grid, kernel_width=kernel_width)
        points.append(p)
        connectivities.append(c)
        splats.append(s)

    points = torch.stack(points)
    connectivities = torch.stack(connectivities)
    splats = torch.stack(splats)

    splats = (splats - torch.mean(splats)) / torch.std(splats)

    return (points[:number_of_meshes_train],
            connectivities[:number_of_meshes_train],
            splats[:number_of_meshes_train],
            points[number_of_meshes_train:],
            connectivities[number_of_meshes_train:],
            splats[number_of_meshes_train:])