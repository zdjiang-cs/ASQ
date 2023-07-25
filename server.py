__author__ = 'zdjiang'

import argparse
import asyncio
import functools
import socket
import pickle
import os
from functools import singledispatch
import asyncio
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F
import copy
import time
import random

from training_module.config import *
from communication_module.comm_utils import *
from training_module import datasets, models, utils
from training_module.action import ServerAction


parser = argparse.ArgumentParser(description='FedMP')
parser.add_argument('--model_type', type=str, default='CNN')
parser.add_argument('--dataset_type', type=str, default='MNIST')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--data_pattern', type=int, default=6)
parser.add_argument('--alpha', type=float, default=200)
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--local_updates', type=int, default=1)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.epoch = args.epoch
    common_config.lr = args.lr
    common_config.local_iters = args.local_updates
    device = torch.device("cuda" if common_config.use_cuda and torch.cuda.is_available() else "cpu")

    worker_idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    comp_time = list()
    comm_time = list()
    epoch_time = list()
    algorithm_time = 0
    total_algorithm_time = 0
    total_time = 0

    client_port = [47016, 47017, 47019, 47020, 47021, 47022, 47023, 47024, 47025, 47026]
    master_port = [57016, 57017, 57019, 57020, 57021, 57022, 57023, 57024, 57025, 57026]

    for idx, worker_idx in enumerate(worker_idx_list):
        custom = dict()
        common_config.worker_list.append(
            Worker(config=ClientConfig(idx=worker_idx,
                                       master_ip_addr=socket.gethostbyname(socket.gethostname()),
                                       action=ClientAction.LOCAL_TRAINING,
                                       custom=custom),
                   ip_addr=WORKER_IP_LIST[idx],
                   master_port=master_port[idx],
                   client_port=client_port[idx]
                   )
        )
    worker_num = len(common_config.worker_list)
    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    global_model = global_model.to(device)
    vm_models = [global_model for vm_idx in range(len(common_config.worker_list))]
    init_para = global_model
    model_tuple = models.Net2Tuple(global_model)
    para_nums = torch.nn.utils.parameters_to_vector(global_model.parameters()).nelement()

    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    train_data_partition, test_data_partition = utils.partition_data(common_config.dataset_type, args.data_pattern)

    for worker_idx, worker in enumerate(common_config.worker_list):
        worker.config.para = init_para
        worker.config.model = model_tuple
        worker.config.custom["dataset_type"] = common_config.dataset_type
        worker.config.custom["model_type"] = common_config.model_type
        worker.config.custom["batch_size"] = common_config.batch_size
        worker.config.custom["learn_rate"] = common_config.learn_rate
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)
        worker.connection = connect_get_socket("MASTER_IP", worker.master_port)

    test_loader = utils.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    global_para = dict(global_model.named_parameters())

    action_queue = Queue()

    for epoch_idx in range(1, 1 + common_config.epoch):
        if epoch_idx != 1:
            start_time = time.time()
            for idx, worker in enumerate(common_config.worker_list):
                vm_models[idx] = copy.deepcopy(global_model)
                worker.config.para = vm_models[idx]
                worker.config.model = models.Net2Tuple(vm_models[idx])
                worker.config.epoch_num = epoch_idx
            algorithm_time = time.time() - start_time

        action_queue.put(ServerAction.SEND_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)

        action_queue.put(ServerAction.GET_STATES)
        ServerAction().execute_action(action_queue.get(), common_config.worker_list)

        with torch.no_grad():
            for idx, worker in enumerate(common_config.worker_list):
                received_para = worker.config.para
                indices = worker.config.indices

                received_para.to(device)
                restored_model = torch.zeros(para_nums).to(device)
                restored_model[indices] = received_para

                worker.config.para = restored_model.data

            global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
            global_para = utils.aggregate_model_with_memory(global_para, common_config.worker_list)
            torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())


        global_model = global_model.to(device)
        global_model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = global_model(data)
                loss_func = nn.CrossEntropyLoss(reduction='sum')
                test_loss += loss_func(output, target).item()
                pred = output.argmax(1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()
                correct += batch_correct

        test_loss /= len(test_loader.dataset)
        test_accuracy = np.float(1.0 * correct / len(test_loader.dataset))

        comp_time.clear()
        comm_time.clear()
        epoch_time.clear()

        for idx, worker in enumerate(common_config.worker_list):
            comp_time.append(worker.config.train_time)
            comm_time.append(worker.config.download_time + worker.config.upload_time)
            true_time = worker.config.train_time + worker.config.download_time + worker.config.upload_time
            epoch_time.append(true_time)

        end_time = time.time()
        total_algorithm_time = total_algorithm_time + algorithm_time
        total_time = total_time + max(epoch_time)

        common_config.recoder.add_scalar('Accuracy/time', test_accuracy, total_time)
        common_config.recoder.add_scalar('Accuracy/epoch', test_accuracy, epoch_idx)
        common_config.recoder.add_scalar('Loss/time', test_loss, total_time)
        common_config.recoder.add_scalar('Loss/epoch', test_loss, epoch_idx)

    for worker in common_config.worker_list:
        worker.connection.shutdown(2)
        worker.connection.close()



if __name__ == "__main__":
    main()
