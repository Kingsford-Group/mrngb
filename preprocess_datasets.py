#!/usr/bin/env python3

from utils import *
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from torchvision.transforms import (Compose, ToPILImage, ToTensor, RandomCrop, RandomHorizontalFlip,
                                    ColorJitter, Normalize)


meta_transforms = lambda dataset, ways, samples: [
    NWays(dataset, ways),
    KShots(dataset, samples),
    LoadData(dataset),
    RemapLabels(dataset),
    ConsecutiveLabels(dataset),
]

std_transforms = Compose([
    # ToPILImage(),
    RandomCrop(112, padding=8),
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    RandomHorizontalFlip(),
    ToTensor(),
    lambda x: x / 255.0,
])

unprocessed_datasets = {
    'flower': lambda root, mode, transform:
        l2l.data.MetaDataset(
            l2l.vision.datasets.VGGFlower102(
                root=root,
                mode=mode,
                transform=transform,
                download=True
            )
        ),
    'aircraft': lambda root, mode, transform:
        l2l.data.MetaDataset(
            l2l.vision.datasets.FGVCAircraft(
                root=root,
                mode=mode,
                transform=transform,
                download=True
            )
        ),
    'fungi': lambda root, mode, transform:
        l2l.data.MetaDataset(
            l2l.vision.datasets.FGVCFungi(
                root=root,
                mode=mode,
                transform=transform,
                download=True
            )
        ),
    'birds': lambda root, mode, transform:
        l2l.data.MetaDataset(
            l2l.vision.datasets.CUBirds200(
                root=root,
                mode=mode,
                transform=transform,
                download=True
            )
        ),
}

def standard_preprocess_tasksets(
    task, root='~/data',
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10
):
    assert task in unprocessed_datasets.keys()

    train_dataset = unprocessed_datasets[task](root, 'train', std_transforms)
    valid_dataset = unprocessed_datasets[task](root,'validation', std_transforms)
    test_dataset = unprocessed_datasets[task](root,'test', std_transforms)

    train_transforms = meta_transforms(train_dataset, train_ways, train_samples)
    valid_transforms = meta_transforms(valid_dataset, test_ways, test_samples)
    test_transforms = meta_transforms(test_dataset, test_ways, test_samples)

    _datasets = (train_dataset, valid_dataset, test_dataset)
    _transforms = (train_transforms, valid_transforms, test_transforms)
    return _datasets, _transforms
