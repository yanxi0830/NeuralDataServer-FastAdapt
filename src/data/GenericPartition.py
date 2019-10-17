from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, has_file_allowed_extension
import os
import pickle
import sys


class GenericPartition(torchvision.datasets.VisionDataset):
    """
    Dataset for given image folder, no labels
    """
    def __init__(self, root, partition_dict_path, cluster_idx=0, loader=default_loader, extensions=IMG_EXTENSIONS,
                 transform=None, target_transform=None):
        super(GenericPartition, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        with open(partition_dict_path, 'rb') as f:
            self.partition_dict = pickle.load(f)

        self.cluster_idx = cluster_idx
        self.samples = self.partition_dict[self.cluster_idx]    # list of image paths

        self.loader = loader
        self.extensions = extensions

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_path) where img_path is the absolute file path to image
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagedir', type=str, required=True, default='',
                        help='image folder directory')
    parser.add_argument('--partition', type=str, required=True, default='',
                        help='path to partition file')
    args = parser.parse_args()
    transform224 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = GenericPartition(root=args.imagedir, partition_dict_path=args.partition, cluster_idx=2,
                               transform=transform224)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    for i, (images, img_path) in enumerate(data_loader):
        print(img_path)
        torchvision.utils.save_image(images, 'check_genericPartition.png')
        break
