import torch
import GPyOpt
import math
import PIL
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from classes import *
from utils import *


def ResNet18():
    return ResNet(ResidualBlock)

transform = T.ToTensor()


# load data
NUM_TRAIN = 49000
print_every = 100


data_dir = './data'

cifar10_train = dset.CIFAR10(data_dir, train=True, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10(data_dir, train=True, download=True, transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)


USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Data Augmentation
n, w, h, c = cifar10_train.train_data.shape
new_train = cifar10_train.train_data.reshape(n*w*h, c) / 255
train_means = [new_train[:,0].mean(), new_train[:,1].mean(), new_train[:,2].mean()]
train_stds = [new_train[:,0].std(), new_train[:,1].std(), new_train[:,2].std()]

n, w, h, c = cifar10_test.test_data.shape
new_test = cifar10_test.test_data.reshape(n*w*h, c) / 255
test_means = [new_test[:,0].mean(), new_test[:,1].mean(), new_test[:,2].mean()]
test_stds = [new_test[:,0].std(), new_test[:,1].std(), new_test[:,2].std()]

transform_test = T.Compose([
    T.Normalize(test_means, test_stds),
    T.ToTensor()
])

transform_train = T.Compose([
    T.Normalize(train_means, train_stds),
    T.ColorJitter(hue=.05, saturation=.05),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20, resample=PIL.Image.BILINEAR),
    T.ToTensor()
])

# load data
NUM_TRAIN = 49000
print_every = 100

data_dir = './data'

cifar10_train_aug = dset.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
cifar10_val_aug = dset.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
cifar10_test_aug = dset.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

loader_train_aug = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val_aug = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
loader_test_aug = DataLoader(cifar10_test, batch_size=64)


# Hyperparameters to tune
bounds = [{'name': 'alpha', 'type': 'continuous', 'domain': (0, 10)}]

# GPyOpt procedure
bopt = GPyOpt.methods.BayesianOptimization(train_opt,
                                           domain=bounds,
                                           model_type='GP_MCMC',
                                           acquisition_type='EI_MCMC',
                                           normalize_Y=True,
                                           n_samples=3)
max_iter = 3
bopt.run_optimization(max_iter)

lr_opt = pow(10, -float(bopt.x_opt))
print('optimal learning rate:', lr_opt)


# define and train the network
model = ResNet18()
optimizer = optim.Adam(model.parameters(), lr=lr_opt)

train_part_data(model, optimizer, loader=loader_train_aug, epochs=10)


# report test set accuracy
check_accuracy(loader_test, model)


# save the model
torch.save(model.state_dict(), 'model.pt')


plt.tight_layout()
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

vis_labels = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

for l in vis_labels:
    getattr(model, l).register_forward_hook(get_activation(l))


data, _ = cifar10_test[0]
data = data.unsqueeze_(0).to(device = device, dtype = dtype)

output = model(data)


for idx, l in enumerate(vis_labels):
    act = activation[l].squeeze()
    if idx < 2:
        ncols = 8
    else:
        ncols = 32

    nrows = act.size(0) // ncols
    fig, axarr = plt.subplots(nrows, ncols)
    fig.suptitle(l)


    for i in range(nrows):
        for j in range(ncols):
            axarr[i, j].imshow(act[i * nrows + j].cpu())
            axarr[i, j].axis('off')
