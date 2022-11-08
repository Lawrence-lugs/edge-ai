
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

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
#transformer for converting data to floattensor
transform = transforms.ToTensor()

#load datasets
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)


# %%

#dataloaders? what is this
num_workers = 0
batch_size = 20
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)

# %%

def imshow(img):
    img = img/2 + 0.5
    plt.imshow(np.transpose(img,(1,2,0)))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#%% Print set batch

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

#hmm what's this...
nrows = 5
fig = plt.figure(figsize=(batch_size*2/nrows,nrows*2))
#this is 20 because batch size is 20
for idx in np.arange(batch_size):
    ax = fig.add_subplot(nrows, int(batch_size/nrows), idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

# %%

import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder,self).__init__()
        ## layers ##
        # conv layer (3 --> 16), 3x3 kernel
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        # conv layer (16 --> 4), 3x3 kernel
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # maxpool stride 2, dims halved.
        self.pool = nn.MaxPool2d(2,2)

        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self,x):
        ## encode ##
        # add hidden layers with relu
        # and maxpool
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        ## decode
        # add transpose conv layer, with relu activation
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling)
        x = F.sigmoid(self.t_conv2(x))
        return x

    def encode(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

model = ConvAutoencoder()
print(model)


# %%
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
n_epochs = 20

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    for data in train_loader:
        images, _ = data
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward() #backpropagate
        optimizer.step() #step the weights
        train_loss += loss.item()*images.size(0)
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch,train_loss))

#%% Save Model
batch = next(iter(train_loader))
images, _ = batch
yhat = model(images)

torch.onnx.export(model,images,'cae.onnx','CIFAR IMG','Label')

#%% Plot test images, test reconstruction
dataiter = iter(test_loader)
images, labels = dataiter.next()
# get sample outputs
output = model(images)
# prep images for display
images = images.numpy()

# output is resized into a batch of iages
output = output.view(batch_size, 3, 32, 32)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    ax.set_title(classes[labels[idx]])
fig.suptitle('Reconstructions')
    
# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
fig.suptitle('Originals')

# %% Plot feature maps

def plotinternalmap(layerrep):
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(8,8))
    for idx,image in enumerate(layerrep.detach().numpy()[0]):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        ax.imshow(image)

dataiter = iter(test_loader)
images, labels = dataiter.next()

im2plot = images
imshow(im2plot[0])
im2plot_layer1 = model.pool(F.relu(model.conv1(images)))
im2plot_layer2 = model.pool(F.relu(model.conv2(im2plot_layer1)))
im2plot_layer3 = F.relu(model.t_conv1(im2plot_layer2))
im2plot_layer4 = F.sigmoid(model.t_conv2(im2plot_layer3))

plotinternalmap(im2plot_layer1)
plotinternalmap(im2plot_layer2)
plotinternalmap(im2plot_layer3)
plotinternalmap(im2plot_layer4)

# %%

imshow(output[0])
# %% Load pretrained feature extractor
from torchvision import models
model_ft = models.mobilenet_v2(pretrained=True)

#%%
print(model_ft)

# %%
pcount = sum(p.numel() for p in model_ft.parameters())
print(pcount)

# %%

from prettytable import PrettyTable

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

#%% Save Model
batch = next(iter(train_loader))
images, _ = batch
yhat = model_ft(images)

torch.onnx.export(model_ft,images,'mbv2.onnx','CIFAR IMG','Label')
# %%
from ptflops import get_model_complexity_info

macs, params = get_model_complexity_info(model_ft, (3,32,32), as_strings=False,print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#%% Reshape a filter into a matrix. 
# What happens to pointwise convolutions?
# need to account for the shape of every convolution

import torchvision

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

matricize_conv2dnormactivation(model_ft.features[0])




#%%

for name,parameter in model_ft.named_parameters():
    a = parameter
    try:
        printshape(a)
    except:
        print(name,a)
        break

#%%


# %%
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

print1layerparams(model_ft)

# %%
