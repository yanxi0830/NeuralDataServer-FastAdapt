from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv
from data.GenericFolder import GenericFolder
from data.GenericPartition import GenericPartition

from pdb import set_trace as breakpoint

# Set the paths of the datasets here.
DATASET_ROOT = './datasets/'
_IMAGENET_DATASET_DIR = './datasets/IMAGENET/ILSVRC2012'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None, config=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.config = config
        self.num_imgs_per_cat = num_imgs_per_cat

        self.mean_pix = [0.485, 0.456, 0.406]
        self.std_pix = [0.229, 0.224, 0.225]

        transforms_list = [
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ]
        self.transform = transforms.Compose(transforms_list)
        self.data = GenericFolder(root=config['image_directory'], transform=self.transform, multi_level=True)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, -1

    def __len__(self):
        return len(self.data)


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2))).copy()
    elif rot == 180:  # 90 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270:  # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1, 0, 2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)

        # if in unsupervised mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
        def _load_function(idx):
            idx = idx % len(self.dataset)
            img0, _ = self.dataset[idx]
            rotated_imgs = [
                self.transform(img0),
                self.transform(rotate_img(img0, 90)),
                self.transform(rotate_img(img0, 180)),
                self.transform(rotate_img(img0, 270))
            ]

            rotation_labels = torch.LongTensor([0, 1, 2, 3])
            return torch.stack(rotated_imgs, dim=0), rotation_labels

        def _collate_fun(batch):
            batch = default_collate(batch)
            assert (len(batch) == 2)
            batch_size, rotations, channels, height, width = batch[0].size()
            batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
            batch[1] = batch[1].view([batch_size * rotations])
            return batch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
