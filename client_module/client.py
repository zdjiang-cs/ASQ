__author__ = 'zdjiang'

import sys
import time
import socket
import pickle
import argparse
import asyncio
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from config import ClientConfig
from client_comm_utils import *
from training_utils import MyNet, train, test
import datasets, models
import utils

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--listen_port', type=int, default=47000, metavar='N',
                    help='Port used to listen msg from master')
parser.add_argument('--master_listen_port', type=int, default=57000, metavar='N',
                    help='')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--min_lr', type=float, default=0.0001)
parser.add_argument('--decay_rate', type=float, default=0.98)
parser.add_argument('--local_iters', type=int, default=1)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--enable_vm_test', action="store_true", default=True)
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--pattern_idx', type=int, default=0)
parser.add_argument('--tx_num', type=int, default=1)

args = parser.parse_args()

MASTER_IP = args.master_ip
LISTEN_PORT = args.listen_port
MASTER_LISTEN_PORT = args.master_listen_port

device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


def main():
    client_config = ClientConfig(
        idx=args.idx,
        master_ip_addr=args.master_ip,
        action=""
    )

    connection = connect_send_socket(MASTER_IP, MASTER_LISTEN_PORT)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    task = asyncio.ensure_future(get_init_config(client_config, connection))
    tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    
    train_dataset, test_dataset = datasets.load_datasets(client_config.custom["dataset_type"])
    train_loader = utils.create_dataloaders(train_dataset,
                                            batch_size=client_config.custom["batch_size"],
                                            selected_idxs=client_config.custom["train_data_idxes"])
    test_loader = utils.create_dataloaders(test_dataset,
                                           batch_size=128,
                                           selected_idxs=client_config.custom["test_data_idxes"],
                                           shuffle=False)

    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                local_training(client_config, connection, train_loader, test_loader)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()


async def local_training(config, conn, train_loader, test_loader):
    model = config.para
    model = model.to(device)
    vm_lr = np.max((args.decay_rate ** (config.epoch_num - 1) * config.custom["learn_rate"], args.min_lr))
    optimizer = optim.SGD(model.parameters(), lr=vm_lr)

    old_para = torch.nn.utils.parameters_to_vector(model.parameters()).clone().detach()

    start_time = time.time()
    train(args, config, model, device, train_loader, test_loader, optimizer, config.epoch_num)
    config.train_time = time.time() - start_time

    local_para = torch.nn.utils.parameters_to_vector(model.parameters()).clone().detach()

    memory_para, compressed_para, indices = utils.compress_model_top_with_memory(local_para, old_para, memory_para, config.compression_ratio)
    
    q_compressed_para = utils.QSGD_com_and_decom(compressed_para, config.quantization_level)

    config.model = models.Net2Tuple(model)
    config.para = q_compressed_para
    config.indices = indices

    config.upload_time = time.time()
    send_data_socket(config, conn)
    
    config_received = get_data_socket(conn)
    config_received.download_time = time.time() - config_received.download_time
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)


async def get_init_config(config, conn):

    config_received = get_data_socket(conn)
    config_received.download_time = time.time() - config_received.download_time

    for k, v in config_received.__dict__.items():
        setattr(config, k, v)


if __name__ == '__main__':
    main()
