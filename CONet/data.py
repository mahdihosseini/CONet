from pathlib import Path

import torchvision.transforms as transforms

import torchvision
import torch
from datasets import ImageNet, TinyImageNet
from utils import Cutout


def get_data(root: Path, dataset: str, mini_batch_size: int, cutout = False, cutout_length = 16):
    train_loader = None
    test_loader = None
    if dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        if cutout:
            print ("Using cutout!")
            transform_train.transforms.append(Cutout(cutout_length))
        else:
            print("Not using cutout!")

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[
                    x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False,
            num_workers=4, pin_memory=True)
    elif dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        if cutout:
            print ("Using cutout!")
            transform_train.transforms.append(Cutout(cutout_length))
        else:
            print("Not using cutout!")

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False,
            num_workers=4, pin_memory=True)
    elif dataset == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(  # New ImageNet
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),  # New
            transforms.CenterCrop(224),  # New  ImageNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        """
        trainset = torchvision.datasets.ImageNet(
            root=str(root), train=True, download=False,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size, shuffle=True, num_workers=4,
            pin_memory=True)
        testset = torchvision.datasets.ImageNet(
            root=str(root), train=False, download=False,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True)
        """

        trainset = ImageNet(
            root=str(root), split='train', download=None,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        testset = ImageNet(
            root=str(root), split='val', download=None,
            transform=transform_test)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    elif dataset == 'COCO':
        ...
    return train_loader, test_loader

