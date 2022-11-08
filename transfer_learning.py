#%%
from re import M
from typing import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torchvision
import time, copy

import law_lib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transformer for converting data to floattensor
transform = transforms.ToTensor()

#load datasets
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

#dataloaders
num_workers = 0
batch_size = 20
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)

# Load models
mbv2 = models.mobilenet_v2(pretrained=True)
#resnet = models.resnet18(pretrained=True)

# this mbv2 has the following structure (baseline, actually):
# [t,c,n,s], [expand,output channels,number of repetitions,stride of first block]
# [1,16,1,1]
# [6,24,2,1]
# [6,32,3,2]
# [6,64,4,2]
# [6,96,3,1]
# [6,160,3,2]
# [6,320,1,1]

# Freeze it
# for param in mbv2.parameters():
#     param.requires_grad = False

# Unfreeze to some depth
# for param in mbv2.features[14:].parameters():
#     param.requires_grad = True


mbv2.features[0][0].stride=(1,1)
mbv2.features[2].conv[1][0].stride=(1,1)

# Freeze it
for param in mbv2.parameters():
    param.requires_grad = False
    
# replace the mbv2 classifier
# new modules are requires_grad = True by default
num_ftrs = mbv2.classifier[1].in_features
mbv2.classifier = nn.Linear(num_ftrs,10)


mbv2 = mbv2.to(device)
criterion = nn.CrossEntropyLoss()


#%% define training algorithm

# Refitted training function for CIFAR and mbv2
def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    accuracies = []
    accuracies.append(test_model(model))

    # deep copy is like np.copy, avoids references.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-'*10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            #send inputs and labels to CUDA?
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                #softmax, give prediction
                _, preds = torch.max(outputs,1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        model.eval()
        scheduler.step()
        accuracies.append(test_model(model))
    
    return model,accuracies

def test_model(net):
    total_correct = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _,preds = torch.max(outputs,1)
        total_correct += torch.sum(preds == labels.data)
    acc = 100*total_correct/len(test_data)
    print(f'Acc: {acc}')
    return acc 

def visualize_model(model, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        test_iter = enumerate(test_loader)
        i=0
        max = np.random.randint(100)
        for i, (inputs, labels) in test_iter:
            if i < max:
                i+=1
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {classes[preds[j]]}')
                plt.imshow(inputs.cpu().data[j].permute(1,2,0))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def train_ae(model,criterion,optimizer,scheduler,num_epochs=25):
    losses = []
    for epoch in range(1, num_epochs-1):
        train_loss = 0.0
        for inputs,labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward() #backpropagate
            optimizer.step() #step the weights
            train_loss += loss.item()*inputs.size(0)
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch,train_loss))
        losses.append(train_loss)
    return model,losses

def remove_sequential(network):
    all_layers = []
    for layer in network.children():
        if list(layer.children()) == []: # if leaf node, add it to list
            all_layers.append(layer)
        else:
            all_layers.extend(remove_sequential(layer))
    return all_layers

def print_model_layertypes(layerset):
    layertypes = []
    for layer in layerset:
        if type(layer) not in layertypes:
            layertypes.append(type(layer))
    print(layertypes)

def matricize_conv(layer):
    mat = torch.flatten(layer.weight,start_dim=1)
    mat = mat.cpu().detach().numpy()
    return mat

def matricize_gconv(layer):
    matin = matricize_conv(layer)
    g = layer.groups
    if g == 1:
        return matin
    kernel_shape = np.shape(matin)[1]
    kernels_per_group = int(np.shape(matin)[0]/g)
    matin = np.reshape(matin,(g,kernels_per_group,kernel_shape))
    mat = []
    for k in np.arange(g):
        mat.append(matin[k])
    return np.array(mat)

def matricize_linear(layer):
    mat = layer.weight
    mat = mat.cpu().detach().numpy()
    mat = np.append(mat,layer.bias.cpu()[:,None],axis=1)
    return mat

def apply_bn(mat,bnlayer):
    bnlayer = bnlayer.cpu()
    a,b,c=0,0,0
    mat2 = np.copy(mat)
    if mat.ndim == 3:
        print(mat.shape)
        a,b,c = mat.shape
        mat = np.reshape(mat,(a*b,c))
    for i,ch in enumerate(mat):
        mat[i] = bnlayer.weight[i] * mat[i]
    mat = np.append(mat,bnlayer.bias[:,None],axis=1)
    if mat2.ndim == 3:
        mat = np.reshape(mat,(a,b,c+1))
    return mat
        
def matricize_model(model):
    layer_list = remove_sequential(model)
    mat_set = []
    prev_mat = 0
    for i,layer in enumerate(layer_list):
        if type(layer)==nn.Conv2d:
            mat_set.append(matricize_gconv(layer))
        if type(layer)==nn.BatchNorm2d:
            # apply to last seen convolution
            mat_set[-1] = apply_bn(mat_set[-1],layer)
        if type(layer)==nn.Linear:
            mat_set.append(matricize_linear(layer))
        #skip activations
    return mat_set

def get_shapes(flatmodel):
    model_shapes = []
    for mat in flatmodel:
        model_shapes.append(mat.shape)
    return model_shapes

mbv2_flat = matricize_model(mbv2)

mbv2_shapes = get_shapes(mbv2_flat)
print(mbv2_shapes)

#%%
def get_max_dims(model_shapes):
    amax = 0
    bmax = 0
    scale = 100
    for mat in model_shapes:
        a = mat[0+(len(mat) == 3)]
        b = mat[1+(len(mat) == 3)]
        if a > amax: amax = a
        if b > bmax: bmax = b
    amax/=scale
    bmax/=scale
    if bmax < 3:
        bmax*=3
    if amax < 3:
        amax*=3
    return bmax,amax

def remove_groups(model_flat):
    out = []
    for mat in model_flat:
        if mat.ndim == 3:
            s = mat.shape
            # insert nan between every row
            matt = np.reshape(mat,(s[0]*s[1],s[2]))
            newim = np.zeros((s[0]*s[1]*2,s[2]))
            newim[:] = np.nan
            newim[::2] = matt
            out.append(newim)
        else:
            out.append(mat)
    return out

def put_max_on_top(imset):
    set = np.copy(imset)
    max = 0
    imax = 0
    for i,mat in enumerate(imset):
        if mat.shape[0] + mat.shape[1] > max:
            max = mat.shape[0] + mat.shape[1]
            imax = i
    set[[0,imax]]=set[[imax,0]]
    print(max)
    return set

# Can probably add a column pertaining to the xdim and then sort by that, then remove the column.
def sort_plotgroups_byxdim(imset):
    imset = np.array(imset)
    set = np.copy(imset)
    for i,mat in enumerate(imset):
        set = put_max_on_top(set)
        imset[i]=set[0]
        set = set[1:]
    return imset

def get_diag_from_model(model_flat):
    model_shapes = get_shapes(model_flat)
    toplot = remove_groups(model_flat)
    n_subplots = len(model_shapes)
    sub_dim = np.sqrt(n_subplots).astype(int)+1
    colmax,rowmax = get_max_dims(model_shapes)
    print(colmax*sub_dim,rowmax*sub_dim)
    plt.figure(figsize=(colmax*sub_dim,rowmax*sub_dim),dpi=5000)
    fig, axs = plt.subplots(nrows=sub_dim,ncols=sub_dim,sharex=True,sharey=True)
    for n,mat in enumerate(toplot):
        ax = axs[int(n/sub_dim),n%sub_dim]
        ax.axis('off')
        ax.imshow(mat)

import matplotlib.cm as cm

def get_diag_each(model_flat):
    toplot = model_flat
    cmap = cm.get_cmap().copy()
    cmap.set_bad(color='white')
    plt.register_cmap('mymap',cmap)

    num_iter = int(len(toplot)/10)
    remainder = len(toplot)%10
    for g in np.arange(num_iter+1):
        fig_size = get_max_dims(get_shapes(toplot[10*g:10*g+10]))
        print(fig_size)
        plt.figure(figsize=(fig_size))
        for i,mat in enumerate(toplot[10*g:10*g+10]):
            plt.subplot(5,2,i+1)
            plt.imshow(mat,cmap='mymap')
            plt.axis('off')
            plt.title(f'{mat.shape}')

# get_diag_from_model(mbv2_flat)
# get_diag_each(mbv2_flat)
a = remove_groups(mbv2_flat)
b = sort_plotgroups_byxdim(a)
get_diag_each(b)

#%%

class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
        ## layers ##
        self.enc = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2,2)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(4,16,2,stride=2),
            nn.ReLU6(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16,3,2,stride=2),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.enc(x)
        x = self.dec(x)
        return x

my = mynet()

#%%

#stochastic gradient descent
optimizer_ft = optim.Adam(mbv2.parameters(),lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

tled_model,acc = train_model(mbv2,criterion,optimizer_ft,exp_lr_scheduler,num_epochs=100)

#%% AE Training

my = my.to(device)
criterion = nn.BCELoss()

#stochastic gradient descent
optimizer_ft = optim.Adam(my.parameters(),lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

my_trained,acc = train_ae(my,criterion,optimizer_ft,exp_lr_scheduler,num_epochs=25)

# %%

torch.save(tled_model.state_dict(),'models/mbv2_9144')

# %%

def plotacc(acc):
    accs = torch.Tensor(acc).cpu()
    accs = accs[1:]
    plt.plot(accs)
    plt.xlabel('Epoch')
    plt.ylabel('CIFAR-10 Accuracy')

plotacc(acc)

# %%
mbv2.load_state_dict(torch.load('models/mbv2_863'))

#%%
mbv2_layers = remove_sequential(mbv2)
print(mbv2_layers)
# %%
