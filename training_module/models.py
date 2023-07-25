import re
import torch.nn.functional as F
import torch.nn as nn


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 64)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def create_model_instance(dataset_type, model_type):
    if dataset_type == 'MNIST':
        if model_type == 'CNN':
            model = MNIST_Net()
        else:
            pass
    return model


def Seq2Tup(sequen):
    lst = []
    for tup in sequen._modules.items():
        lst.append(tup)
    return lst


def ExtractParam(string, num):
    str_num_list = re.findall(r'[0-9]+\.?[0-9]*', string[string.find('('):])
    if num == 1:
        return float(str_num_list[0])
    num_list = [0] * num
    idx1 = 0
    while idx1 < num:
        if num == 5 and idx1 == 2:
            if str_num_list[2] == str_num_list[3]:
                num_list[2] = int(str_num_list[2])
            else:
                num_list[2] = int(str_num_list[2]), int(str_num_list[3])
            if str_num_list[4] == str_num_list[5]:
                num_list[3] = int(str_num_list[4])
            else:
                num_list[3] = int(str_num_list[4]), int(str_num_list[5])
            if len(str_num_list) == 8:
                if str_num_list[6] == str_num_list[7]:
                    num_list[4] = int(str_num_list[7])
                else:
                    num_list[4] = int(str_num_list[6]), int(str_num_list[7])
            idx1 = 5
        else:
            num_list[idx1] = int(str_num_list[idx1])
            idx1 += 1
    return tuple(num_list)


def FunType(string):
    if string.find('BatchNorm') != -1:
        return 'Batchnorm'
    elif string.find('ReLU') != -1:
        return 'Relu'
    elif string.find('Sigmoid') != -1:
        return 'Sigmoid'
    elif string.find('MaxPool') != -1:
        return ExtractParam(string, 3)
    elif string.find('Dropout') != -1:
        return ExtractParam(string, 1)
    elif string.find('Softmax') != -1:
        return 'Softmax'


def Net2Tuple(net):
    tmp = nn.Sequential()
    net_list = []
    net_param_list = []
    for item in net._modules.items():
        if isinstance(item[1], type(tmp)):
            net_list += Seq2Tup(item[1])
        else:
            net_list.append(item)

    idx = 0
    while idx < len(net_list):
        layer = []
        layer.append(net_list[idx][0])
        tostr = str(net_list[idx][1])
        if tostr.find('Conv') != -1:
            layer.append('Conv')
            idx += 1
            layer.append(ExtractParam(tostr, 5))
            while (idx < len(net_list) and
                   str(net_list[idx][1]).find('Linear') == -1 and
                   str(net_list[idx][1]).find('Conv') == -1):
                if str(net_list[idx][1]).find('MaxPool') != -1:
                    layer.append('Maxpool')
                layer.append(FunType(str(net_list[idx][1])))
                idx += 1
            net_param_list.append(tuple(layer))
        else:
            layer.append('FC')
            idx += 1
            layer.append(ExtractParam(tostr, 2))
            while idx < len(net_list) and str(net_list[idx][1]).find('Linear') == -1:
                if str(net_list[idx][1]).find('Dropout') != -1:
                    layer.append('Dropout')
                layer.append(FunType(str(net_list[idx][1])))
                idx += 1
            net_param_list.append(tuple(layer))
    return net_param_list