import os
import time
import math
import numpy as np
import numpy.ma as ma
import random
import re

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import datasets


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:>3}m {:2.0f}s'.format(m, s)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_vm_by_id(id_str):
    vm_instance = sy.VirtualWorker(hook, id=id_str)

    return vm_instance


def create_vm(vm_num=2):
    vm_list = list()
    for vm_idx in range(vm_num):
        vm_id = 'vm' + str(vm_idx)
        vm_instance = create_vm_by_id(vm_id)
        vm_list.append(vm_instance)
    return vm_list


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class RandomPartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in partition_sizes:
            part_len = round(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        selected_idxs = self.partitions[partition]
        return selected_idxs


class LabelwisePartitioner(object):

    def __init__(self, data, partition_sizes, seed=2020):
        self.data = data
        self.partitions = [list() for _ in range(len(partition_sizes[0]))]
        rng = random.Random()
        rng.seed(seed)

        label_indexes = list()
        class_len = list()

        for class_idx in range(len(data.classes)):
            label_indexes.append(list(np.where(data.targets == class_idx)[0]))
            class_len.append(len(label_indexes[class_idx]))
            rng.shuffle(label_indexes[class_idx])

        for class_idx in range(len(data.classes)):
            begin_idx = 0
            for vm_idx, frac in enumerate(partition_sizes[class_idx]):
                end_idx = begin_idx + round(frac * class_len[class_idx])
                self.partitions[vm_idx].extend(label_indexes[class_idx][begin_idx:end_idx])
                begin_idx = end_idx

    def use(self, partition):
        selected_idxs = self.partitions[partition]
        return selected_idxs


def create_dataloaders(dataset, batch_size, selected_idxs=None, shuffle=True, pin_memory=True, num_workers=4):
    if selected_idxs is None:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                pin_memory=pin_memory,
                                num_workers=num_workers)
    else:
        partition = Partition(dataset, selected_idxs)
        dataloader = DataLoader(partition,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                pin_memory=pin_memory,
                                num_workers=num_workers)

    return dataloader


def create_bias_selected_data(args, selected_idxs, dataset):
    if not isinstance(dataset.targets, np.ndarray):
        dataset.targets = np.array(dataset.targets)

    if not isinstance(dataset.data, np.ndarray):
        dataset.data = np.array(dataset.data)

    indices = np.isin(dataset.targets, selected_idxs).astype("bool")
    selected_targets = dataset.targets[indices]
    selected_data = dataset.data[indices]

    return np.float32(selected_data), np.int64(selected_targets)


def create_bias_loader(args, kwargs, is_train, dataset, selected_idxs):
    vm_loaders = list()
    for vm_idx in range(0, args.vm_num):
        selected_data, selected_targets = create_bias_selected_data(
            args, selected_idxs[vm_idx], dataset)
        if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
            selected_data = np.transpose(selected_data, (0, 3, 1, 2))

        data_len = len(selected_data)

        data_transform = datasets.load_default_transform(args.dataset_type)
        vm_dataset_instance = datasets.VMDataset(selected_data, selected_targets, data_transform)

        if is_train:
            vm_loader_instance = DataLoader(vm_dataset_instance,
                                            shuffle=True,
                                            batch_size=args.batch_size,
                                            **kwargs)
        else:
            vm_loader_instance = DataLoader(vm_dataset_instance,
                                            shuffle=False,
                                            batch_size=args.test_batch_size,
                                            **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders


def create_bias_federated_loader(args, kwargs, vm_list, is_train, dataset, selected_idxs):
    vm_loaders = list()
    for vm_idx in range(0, args.vm_num):
        selected_data, selected_targets = create_bias_selected_data(
            args, selected_idxs[vm_idx], dataset)
        if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
            selected_data = np.transpose(selected_data, (0, 3, 1, 2))

        data_len = len(selected_data)

        data_transform = datasets.load_default_transform(args.dataset_type)
        vm_dataset_instance = datasets.VMDataset(
            selected_data, selected_targets, data_transform).federate([vm_list[vm_idx]])

        if is_train:
            vm_loader_instance = sy.FederatedDataLoader(vm_dataset_instance,
                                                        shuffle=True,
                                                        batch_size=args.batch_size,
                                                        **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader(vm_dataset_instance,
                                                        shuffle=False,
                                                        batch_size=args.test_batch_size,
                                                        **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders


def create_random_selected_data(args, num_data, dataset):
    data_idx = [i for i in range(len(dataset.data))]
    random.shuffle(data_idx)

    select_data_idx = data_idx[:num_data]

    selected_targets = dataset.targets[select_data_idx]
    selected_data = dataset.data[select_data_idx]

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        selected_data = np.transpose(selected_data, (0, 3, 1, 2))

    return np.float32(selected_data), np.int64(selected_targets)


def create_segment_selected_data(args, begin_idx, end_idx, dataset):
    selected_targets = dataset.targets[begin_idx:end_idx]
    selected_data = dataset.data[begin_idx:end_idx]

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        selected_data = np.transpose(selected_data, (0, 3, 1, 2))

    return np.float32(selected_data), np.int64(selected_targets)


def create_labelwise_selected_data(args, label_wise_data, label_wise_targets):
    class_num = len(label_wise_targets)

    targets_shape = list(label_wise_targets[0][0].shape)
    targets_shape[0] = 0
    init_targets_shape = tuple(targets_shape)
    selected_targets = np.empty(init_targets_shape)

    data_shape = list(label_wise_data[0][0].shape)
    data_shape[0] = 0
    init_data_shape = tuple(data_shape)
    selected_data = np.empty(init_data_shape)

    for idx in range(class_num):
        slice_idxs = list(range(len(label_wise_targets[idx])))
        random.shuffle(slice_idxs)
        selected_targets = np.concatenate(
            (selected_targets, label_wise_targets[idx].pop(slice_idxs[0])), axis=0)
        selected_data = np.concatenate(
            (selected_data, label_wise_data[idx].pop(slice_idxs[0])), axis=0)

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        selected_data = np.transpose(selected_data, (0, 3, 1, 2))

    return np.float32(selected_data), np.int64(selected_targets)


def create_random_loader(args, kwargs, tx2_idx, num_data, is_train, dataset):
    data_len = len(dataset.targets)

    selected_data, selected_targets = create_random_selected_data(
        args, num_data, dataset)

    data_transform = datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = datasets.VMDataset(selected_data, selected_targets, data_transform)
    if is_train:
        vm_loader = DataLoader(vm_dataset_instance,
                               shuffle=True,
                               batch_size=args.batch_size,
                               **kwargs)
    else:
        vm_loader = DataLoader(vm_dataset_instance,
                               shuffle=False,
                               batch_size=args.test_batch_size,
                               **kwargs)

    return vm_loader


def create_segment_loader(args, kwargs, num_tx2, tx2_idx, is_train, dataset):
    data_len = len(dataset.targets)
    inter_num = np.int32(np.floor(data_len / num_tx2))
    begin_idx = tx2_idx * inter_num
    if tx2_idx != num_tx2:
        end_idx = (tx2_idx + 1) * inter_num
    else:
        end_idx = data_len

    selected_data, selected_targets = create_segment_selected_data(
        args, begin_idx, end_idx, dataset)

    data_transform = datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = datasets.VMDataset(selected_data, selected_targets, data_transform)

    if is_train:
        vm_loader = DataLoader(vm_dataset_instance,
                               shuffle=True,
                               batch_size=args.batch_size,
                               **kwargs)
    else:
        vm_loader = DataLoader(vm_dataset_instance,
                               shuffle=False,
                               batch_size=args.batch_size,
                               **kwargs)

    return vm_loader


def create_segment_federated_loader(args, kwargs, vm_list, is_train, dataset):
    vm_loaders = list()
    data_len = len(dataset.targets)
    inter_num = np.int32(np.floor(data_len / len(vm_list)))
    for vm_idx in range(len(vm_list)):
        begin_idx = vm_idx * inter_num
        if vm_idx != len(vm_list) - 1:
            end_idx = (vm_idx + 1) * inter_num
        else:
            end_idx = data_len

        selected_data, selected_targets = create_segment_selected_data(
            args, begin_idx, end_idx, dataset)

        data_transform = datasets.load_default_transform(args.dataset_type)

        vm_dataset_instance = datasets.VMDataset(
            selected_data, selected_targets, data_transform).federate([vm_list[vm_idx]])
        if is_train:
            vm_loader_instance = sy.FederatedDataLoader(vm_dataset_instance,
                                                        shuffle=True,
                                                        batch_size=args.batch_size,
                                                        **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader(vm_dataset_instance,
                                                        shuffle=False,
                                                        batch_size=args._test_batch_size,
                                                        **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders


def create_labelwise_federated_loader(args, kwargs, vm_list, is_train, dataset, partition_ratios):
    vm_loaders = list()
    class_num = len(dataset.classes)
    label_wise_data = [[] for idx in range(class_num)]
    label_wise_targets = [[] for idx in range(class_num)]
    targets_array = np.array(dataset.targets)
    for c_idx in range(class_num):
        label_targets = targets_array[targets_array == c_idx]
        label_data = dataset.data[targets_array == c_idx]
        label_item_num = len(label_targets)
        begin_idx = 0
        for pr_idx in range(len(partition_ratios)):
            if pr_idx == len(partition_ratios) - 1:
                end_idx = label_item_num
            else:
                end_idx = np.min(
                    (begin_idx + np.int32(np.floor(label_item_num * partition_ratios[pr_idx])), label_item_num))
            label_wise_targets[c_idx].append(label_targets[begin_idx:end_idx])
            label_wise_data[c_idx].append(label_data[begin_idx:end_idx])
            begin_idx = end_idx

    for vm_idx in range(len(vm_list)):
        selected_data, selected_targets = create_labelwise_selected_data(
            args, label_wise_data, label_wise_targets)

        data_transform = datasets.load_default_transform(args.dataset_type)

        vm_dataset_instance = datasets.VMDataset(
            selected_data, selected_targets, data_transform).federate([vm_list[vm_idx]])

        if is_train:
            vm_loader_instance = sy.FederatedDataLoader(vm_dataset_instance,
                                                        shuffle=True,
                                                        batch_size=args.batch_size,
                                                        **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader(vm_dataset_instance,
                                                        shuffle=False,
                                                        batch_size=args.test_batch_size,
                                                        **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders


def create_test_loader(args, kwargs, test_dataset):
    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        test_data = np.transpose(test_dataset.data, (0, 3, 1, 2))
    else:
        test_data = test_dataset.data

    test_data = torch.tensor(np.float32(test_data))
    test_targets = torch.tensor(np.int64(test_dataset.targets))
    test_loader = DataLoader(TensorDataset(test_data, test_targets),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return test_loader


def create_server_test_loader(args, kwargs, test_dataset):
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)

    if not isinstance(test_dataset.data, np.ndarray):
        test_dataset.data = np.array(test_dataset.data)

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        test_dataset.data = np.transpose(
            test_dataset.data, (0, 3, 1, 2))

    data_transform = datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = datasets.VMDataset(np.float32(test_dataset.data),
                                             np.int64(test_dataset.targets),
                                             data_transform)

    test_loader = DataLoader(vm_dataset_instance,
                             shuffle=False,
                             batch_size=args.test_batch_size,
                             **kwargs)

    return test_loader


def create_ps_test_loader(args, kwargs, vm_instance, test_dataset):
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)

    if not isinstance(test_dataset.data, np.ndarray):
        test_dataset.data = np.array(test_dataset.data)

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        test_dataset.data = np.transpose(test_dataset.data, (0, 3, 1, 2))

    data_transform = datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = datasets.VMDataset(np.float32(test_dataset.data),
                                             np.int64(test_dataset.targets),
                                             data_transform).federate([vm_instance])

    test_loader = sy.FederatedDataLoader(vm_dataset_instance,
                                         shuffle=False,
                                         batch_size=args.test_batch_size,
                                         **kwargs)

    return test_loader


def create_centralized_train_test_loader(args, kwargs, vm_instance, vm_dataset, is_test=False):
    if not isinstance(vm_dataset.targets, np.ndarray):
        vm_dataset.targets = np.array(vm_dataset.targets)

    if not isinstance(vm_dataset.data, np.ndarray):
        vm_dataset.data = np.array(vm_dataset.data)

    if args.dataset_type == 'FashionMNIST':
        data_transform = None
    else:
        data_transform = datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = datasets.VMDataset(np.float32(vm_dataset.data),
                                             np.int64(vm_dataset.targets),
                                             data_transform).federate([vm_instance])

    if is_test:
        vm_loader = sy.FederatedDataLoader(vm_dataset_instance,
                                           shuffle=False,
                                           batch_size=args.test_batch_size,
                                           **kwargs)
    else:
        vm_loader = sy.FederatedDataLoader(vm_dataset_instance,
                                           shuffle=True,
                                           batch_size=args.batch_size,
                                           **kwargs)

    return vm_loader


def add_model(dst_model, src_model):
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    param1.data + dict_params2[name1].data)
    return dst_model


def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def weight_scale_model(model, weight_dict):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * weight_dict[name])
    return model


def calculate_model_divergence(dst_model, src_model):
    div_dict = dict()
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                tmp_dist = torch.norm(
                    param1.data - dict_params2[name1].data, p=2)
                tmp_norm = torch.norm(dict_params2[name1].data, p=2)
                if tmp_norm == 0.0:
                    div_dict[name1] = 0.0
                else:
                    div_dict[name1] = tmp_dist / tmp_norm
                del tmp_dist
                del tmp_norm

    return div_dict


def compress_model_top_with_memory(local_para, old_para, memory_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        local_para=local_para-old_para+memory_para
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(local_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    restored_model = torch.zeros(send_para.nelement()).to(device)
    restored_model[indices] = select_para
    memory_para=local_para - restored_model
    model_size = select_para.nelement() * 4 / 1024 / 1024
    return memory_para, select_para, indices


def QSGD_com_and_decom(vec, num_bit):
    signs = torch.sign(vec) > 0
    s = 2 ** num_bit
    norm =  torch.max(torch.abs(vec))
    normalized_vec = vec / norm
    scaled_vec = torch.abs(normalized_vec) * s
    l = torch.clamp(scaled_vec, 0, s).type(torch.int32)

    probabilities = scaled_vec - l.type(torch.float32)
    probabilities.cuda()
    r = torch.rand(l.size()).cuda()
    l[:] += (probabilities > r).type(torch.int32)

    scaled_vec = l.type(torch.float32) * (2 * signs.type(torch.float32) - 1) 
    compressed = scaled_vec * norm / s
    return compressed