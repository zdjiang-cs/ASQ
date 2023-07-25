import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class VMDataset(Dataset):

    def __init__(self, data, targets, classes, transform=transforms.ToTensor()):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = int(self.targets[idx])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def load_datasets(dataset_type, data_path="/dataset"):
    transform = load_default_transform(dataset_type)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, transform=transform)

    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_path, train=False, transform=transform)

    elif dataset_type == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_path, train=False, transform=transform)

    elif dataset_type == 'MNIST':
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    return train_dataset, test_dataset


def load_default_transform(dataset_type):
    dataset_transform = []

    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

    elif dataset_type == 'CIFAR100':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif dataset_type == 'FashionMNIST':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_type == 'MNIST':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    return dataset_transform


def load_customized_transform(dataset_type):
    dataset_transform = []

    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

    elif dataset_type == 'CIFAR100':
        dataset_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    elif dataset_type == 'FashionMNIST':
        dataset_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor()
        ])

    elif dataset_type == 'MNIST':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    return dataset_transform
