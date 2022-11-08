
# Code based on https://www.kaggle.com/code/ljlbarbosa/convolution-autoencoder-pytorch
# Written by Lawrence Quizon

#%%
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision
from prettytable import PrettyTable
import time
import copy 

def imshow(img):
    img = img/2 + 0.5
    plt.imshow(np.transpose(img,(1,2,0)))

def plotinternalmap(layerrep):
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(8,8))
    for idx,image in enumerate(layerrep.detach().numpy()[0]):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        ax.imshow(image)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def printshape(param):
    print(f'layer shape:{a.shape}')
    print(f'{a.shape[0]} output channels')
    print(f'{a.shape[1]} input channels')
    print(f'{a.shape[2]} x {a.shape[3]} kernel')

def matricize_conv2dnormactivation(layer):
    if not type(layer) == torchvision.ops.misc.Conv2dNormActivation:
        print(f'input type is {type(layer)}')
        return 1
    print(layer[0].weight.shape)
    mat = torch.flatten(layer[0].weight,start_dim=1)
    mat = mat.detach()
    for i,ch in enumerate(mat.T):
        mat.T[i] = layer[1].weight * mat.T[i] + layer[1].bias
    return mat

def apply_flattened_convolution(input):
    mat = []
    return mat

def print1layerparams(network):
    new = remove_sequential(network)
    print(new)
    for layer in new:
        print(a)
        print(layer.parameters)
        break

def remove_sequential(network):
    all_layers = []
    for layer in network.children():
        print(layer.name)
        if type(layer) == nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer)
        if list(layer.children()) == []: # if leaf node, add it to list
            all_layers.append(layer)
    return all_layers


# %%
