""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import torch
import tarfile
import random
import numpy as np
import pickle
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

from config.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from datasets.transforms import _pil_interp


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


def load_class_map(filename, root=''):
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    else:
        assert False, 'Unsupported class map extension'
    return class_to_idx


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]


def _extract_tar_info(tarfile, class_to_idx=None, sort=True):
    files = []
    labels = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        label = os.path.basename(dirname)
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    tarinfo_and_targets = [(f, class_to_idx[l]) for f, l in zip(files, labels) if l in class_to_idx]
    if sort:
        tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets, class_to_idx


class DatasetTar(data.Dataset):

    def __init__(self, root, load_bytes=False, transform=None, class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isfile(root)
        self.root = root
        with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
            self.samples, self.class_to_idx = _extract_tar_info(tf, class_to_idx)
        self.imgs = self.samples
        self.tarfile = None  # lazy init in __getitem__
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.samples[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = iob.read() if self.load_bytes else Image.open(iob).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename

    def filenames(self, basename=False):
        fn = os.path.basename if basename else lambda x: x
        return [fn(x[0].name) for x in self.samples]


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)


class DatasetTxt(data.Dataset):

    def __init__(self,
                 root,
                 txt_path,
                 load_bytes=False,
                 transform=None,
                 class_map='',
                 return_path=False):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, img_names = self._find_images_and_targets(root, txt_path, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.img_names = img_names
        self.imgs = self.samples  # torchvision ImageFolder compat
        # self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform
        self.return_path = return_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        if self.return_path:
            return img, target, index

        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]

    def _find_images_and_targets(self,
                                 folder,
                                 txt_path,
                                 types=IMG_EXTENSIONS,
                                 class_to_idx=None,
                                 leaf_name_only=True,
                                 sort=True):
        labels = []
        filenames = []
        img_names = []
        with open(os.path.join(folder, txt_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_relpath, class_id = line.strip('\n').split(',')
                filenames.append(os.path.join(folder, img_relpath))
                labels.append(class_id)
                img_names.append(img_relpath.split('/')[-1])
        images_and_targets = [(f, int(l)) for f, l in zip(filenames, labels)]

        # if class_to_idx is None:
        #     # building class index
        #     unique_labels = set(labels)
        #     sorted_labels = list(sorted(unique_labels, key=natural_key))
        #     class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
        # images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
        # if sort:
        #     images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
        return images_and_targets, img_names





class DatasetTxtPkl(data.Dataset):

    def __init__(self,
                 root,
                 txt_path,
                 pkl_path,
                 load_bytes=False,
                 transform=None,
                 class_map='',
                 smoothing=0.1,
                 return_path=False):

        self.smoothing = smoothing
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = self._find_images_and_targets(root, txt_path, class_to_idx=class_to_idx)
        images_pkl, _ = self._find_images_and_targets_pkl(root, pkl_path, class_to_idx)
        images.extend(images_pkl)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.samples_dict = dict()
        for (filename, target) in self.samples:
            self.samples_dict[filename] = target
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform
        self.return_path = return_path

    def update_samples(self, filenames, targets):
        for (filename, target) in zip(filenames, targets):
            self.samples_dict[filename] = target

    def __getitem__(self, index):
        path, target = self.samples[index]
        target = self.samples_dict[path]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        if self.return_path:
            return img, target, path

        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]

    def _find_images_and_targets(self,
                                 folder,
                                 txt_path,
                                 types=IMG_EXTENSIONS,
                                 class_to_idx=None,
                                 leaf_name_only=True,
                                 sort=True):
        labels = []
        filenames = []
        with open(os.path.join(folder, txt_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_relpath, class_id = line.strip('\n').split(',')
                filenames.append(os.path.join(folder, img_relpath))
                labels.append(class_id)
        if class_to_idx is None:
            # building class index
            unique_labels = set(labels)
            self.num_classes = len(unique_labels)
            sorted_labels = list(sorted(unique_labels, key=natural_key))
            class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
        images_and_targets = [(f, self.one_hot(class_to_idx[l])) for f, l in zip(filenames, labels) if l in class_to_idx]
        if sort:
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
        return images_and_targets, class_to_idx

    def _find_images_and_targets_pkl(self, folder, pkl_path, class_to_idx):
        images_and_targets = []
        data = pickle.load(open(os.path.join(folder, pkl_path), 'rb'))
        num_samples = len(data['paths'])
        for i in range(num_samples):
            images_and_targets.append((os.path.join(folder, data['paths'][i]), data['probs'][i]))
        return images_and_targets, class_to_idx

    def one_hot(self, label):
        one_hot_label = np.zeros(self.num_classes, dtype=np.float32)
        one_hot_label[label] = 1
        off_value = self.smoothing / self.num_classes
        on_value = 1. - self.smoothing / self.num_classes
        one_hot_label[one_hot_label == 0] = off_value
        one_hot_label[one_hot_label == 1] = on_value
        return one_hot_label


class DatasetTxtPseudo(data.Dataset):

    def __init__(self,
                 root,
                 txt_path,
                 load_bytes=False,
                 transform=None,
                 class_map='',
                 return_path=False):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = self._find_images_and_targets(root, txt_path, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.samples_dict = dict()
        for (filename, target) in self.samples:
            self.samples_dict[filename] = target
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform
        self.return_path = return_path

    def update_samples(self, filenames, targets):
        for (filename, target) in zip(filenames, targets):
            self.samples_dict[filename] = target

    def __getitem__(self, index):
        path, target = self.samples[index]
        target = self.samples_dict[path]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        if self.return_path:
            return img, target, path

        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]

    def _find_images_and_targets(self,
                                 folder,
                                 txt_path,
                                 types=IMG_EXTENSIONS,
                                 class_to_idx=None,
                                 leaf_name_only=True,
                                 sort=True):
        labels = []
        filenames = []
        with open(os.path.join(folder, txt_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_relpath, class_id = line.strip('\n').split(',')
                filenames.append(os.path.join(folder, img_relpath))
                labels.append(class_id)
        if class_to_idx is None:
            # building class index
            unique_labels = set(labels)
            sorted_labels = list(sorted(unique_labels, key=natural_key))
            class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
        images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
        if sort:
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
        return images_and_targets, class_to_idx


class DatasetTxtGAN(data.Dataset):

    def __init__(self,
                 root,
                 txt_path,
                 txt_gan_path,
                 gan_ratio=0.5,
                 load_bytes=False,
                 transform=None,
                 class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = self._find_images_and_targets(root, txt_path, class_to_idx=class_to_idx)
        gan_images, _ = self._find_images_and_targets(root, txt_gan_path, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.gan_samples = gan_images
        self.gan_ratio = gan_ratio
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.gan_imgs = self.gan_samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if random.uniform(0., 1.0) > self.gan_ratio:
            path, target = self.samples[index]
        else:
            path, target = self.gan_samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]

    def _find_images_and_targets(self,
                                 folder,
                                 txt_path,
                                 types=IMG_EXTENSIONS,
                                 class_to_idx=None,
                                 leaf_name_only=True,
                                 sort=True):
        labels = []
        filenames = []
        with open(os.path.join(folder, txt_path), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_relpath, class_id = line.strip('\n').split(',')
                filenames.append(os.path.join(folder, img_relpath))
                labels.append(class_id)
        if class_to_idx is None:
            # building class index
            unique_labels = set(labels)
            sorted_labels = list(sorted(unique_labels, key=natural_key))
            class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
        images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
        if sort:
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
        return images_and_targets, class_to_idx
