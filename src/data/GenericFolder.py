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
import sys


class GenericFolder(torchvision.datasets.VisionDataset):
    """
    Dataset for given image folder, no labels
    """
    def __init__(self, root, loader=default_loader, extensions=IMG_EXTENSIONS, transform=None, target_transform=None,
                 is_valid_file=None, multi_level=False):
        super(GenericFolder, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.multi_level = multi_level
        samples = self.make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " +
                               self.root + "\nSupported extensions are: " + ",".join(extensions))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_path) where img_path is the absolute file path to image
        """
        path, target = self.samples[index]  # no labeled target => 0
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, path

    def __len__(self):
        return len(self.samples)

    def make_dataset(self, dir, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        d = dir
        if self.multi_level:
            for root, _, fnames in sorted(os.walk(d)):
                for fname in tqdm(sorted(fnames)):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (path, 0)
                        images.append(item)
        else:
            for fname in tqdm(sorted(os.listdir(d))):
                path = os.path.join(dir, fname)
                if is_valid_file(path):
                    item = (path, 0)
                    images.append(item)

        return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagedir', type=str, required=True, default='',
                        help='image folder directory')
    args = parser.parse_args()
    transform224 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = GenericFolder(root=args.imagedir, transform=transform224)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    for i, (images, img_path) in enumerate(data_loader):
        print(img_path)
        torchvision.utils.save_image(images, 'check_genericFolder.png')
        break
