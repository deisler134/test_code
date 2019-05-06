'''
Created on May 3, 2019

    show how to using tensorboard in pytorch (test)
    
        'from torch.utils.tensorboard import SummaryWriter'
        
        ref: https://pytorch.org/docs/stable/tensorboard.html?highlight=tensorboard
    
    note: EXPERIMENTAL code that might change in the future.
    
        should be installable and runable with: 
        
        *   $ pip install tb-nightly  # Until 1.14 moves to the release channel
        *   $ tensorboard --logdir=runs (or  tensorboard --logdir runs)

@author: deisler
'''


import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

# print( type(images) )
# 
# input = (torch.Tensor(1,1,28,28),)
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()

