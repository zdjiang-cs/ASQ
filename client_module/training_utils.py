import torch.nn as nn
import torch
import re
import sys
import time
import numpy as np
import torch.nn.functional as F
import gc
from utils import printer, time_since


def MakeLayers(params_list):
    conv_layers = nn.Sequential()
    fc_layers = nn.Sequential()
    c_idx = p_idx = f_idx = 1
    for param in params_list:
        if param[1] == 'Conv':
            conv_layers.add_module(param[0],
                                   nn.Conv2d(param[2][0], param[2][1], param[2][2], param[2][3], param[2][4]))
            if len(param) >= 4:
                if param[3] == 'Batchnorm':
                    conv_layers.add_module('batchnorm' + str(c_idx), nn.BatchNorm2d(param[2][1]))
                if param[3] == 'Relu' or (param[3] == 'Batchnorm' and param[4] == 'Relu'):
                    conv_layers.add_module('relu' + str(c_idx), nn.ReLU(inplace=True))
                else:
                    conv_layers.add_module('sigmoid' + str(c_idx), nn.Sigmoid())
            if len(param) >= 6:
                if param[3] == 'Batchnorm':
                    if param[5] == 'Maxpool':
                        conv_layers.add_module(
                            'maxpool' + str(p_idx), nn.MaxPool2d(param[6][0], param[6][1], param[6][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool' + str(p_idx), nn.AvgPool2dparam[6][0], param[6][1], param[6][2])
                else:
                    if param[4] == 'Maxpool':
                        conv_layers.add_module(
                            'maxpool' + str(p_idx), nn.MaxPool2d(param[5][0], param[5][1], param[5][2]))
                    else:
                        conv_layers.add_module(
                            'avgpool' + str(p_idx), nn.AvgPool2dparam[5][0], param[5][1], param[5][2])
                p_idx += 1
            c_idx += 1

        else:
            fc_layers.add_module(param[0], nn.Linear(param[2][0], param[2][1]))
            if len(param) >= 4:
                if param[3] == 'Dropout':
                    fc_layers.add_module('dropout', nn.Dropout(param[4]))
                if param[3] == 'Relu' or (param[3] == 'Dropout' and len(param) == 6 and param[5] == 'Relu'):
                    fc_layers.add_module('relu' + str(f_idx), nn.ReLU(inplace=True))
                elif param[3] == 'Sigmoid' or (param[3] == 'Dropout' and len(param) == 6 and param[5] == 'Sigmoid'):
                    fc_layers.add_module('sigmoid' + str(f_idx), nn.Sigmoid())
                elif param[3] == 'Softmax' or (param[3] == 'Dropout' and len(param) == 6 and param[5] == 'Softmax'):
                    fc_layers.add_module('softmax' + str(f_idx), nn.Softmax())
            f_idx += 1
    return conv_layers, fc_layers



def train(args, config, tx2_model, device, tx2_train_loader, tx2_test_loader, tx2_optimizer, epoch):

    vm_start = time.time()
    tx2_model.train()
    train_loss = 0.0
    samples_num = 0
    print("local_iters: ", args.local_iters)
    for iter_idx in range(args.local_iters):
        vm_data, vm_target = next(tx2_train_loader)

        if config.custom["dataset_type"] == 'FashionMNIST' or config.custom["dataset_type"] == 'MNIST':
            if config.custom["model_type"] == 'LR':
                vm_data = vm_data.squeeze(1) 
                vm_data = vm_data.view(-1, 28 * 28)
            else:
                pass
                
        vm_data, vm_target = vm_data.to(device), vm_target.to(device)
        vm_output = tx2_model(vm_data)

        tx2_optimizer.zero_grad()
        
        loss_func = nn.CrossEntropyLoss() 
        vm_loss = loss_func(vm_output, vm_target)
        vm_loss.backward()
        tx2_optimizer.step()

        train_loss += (vm_loss.item() * vm_data.size(0))
        samples_num += vm_data.size(0)
        del vm_data
        del vm_target
        del vm_output
        del vm_loss
    


def test(args, start, model, device, test_loader, epoch):
    model.eval()

    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                if args.model_type == 'LR':
                    data = data.squeeze(1) 
                    data = data.view(-1, 28 * 28)
                else:
                    pass

            if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                if args.model_type == 'LSTM':
                    data = data.view(-1, 32, 32 * 3)                    
                else:
                    pass  

            if args.model_type == 'LSTM':
                hidden = model.initHidden(args.test_batch_size)
                hidden = hidden.send(data.location)
                for col_idx in range(32):
                    data_col = data[:, col_idx, :]
                    output, hidden = model(data_col, hidden)
            else:
                output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
                
            del data
            del target
            del output
            del pred
            del batch_correct

    test_loss /= len(test_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(test_loader.dataset))

    gc.collect()

    return test_loss, test_accuracy
