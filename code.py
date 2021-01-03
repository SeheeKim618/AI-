import sys
sys.path.append('path/of/AI-Advanced-Course')

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/AI-Advanced-Course-master')

import imp

from matplotlib import pyplot as plt

try:
    imp.find_module('jupyterplot')
    from jupyterplot import ProgressPlot
except ImportError:
    !pip install jupyterplot
    from jupyterplot import ProgressPlot
    
    %matplotlib inline
from matplotlib import pyplot as plt
from jupyterplot import ProgressPlot

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets as D
from torchvision import transforms as T

from utils import sample_random_data
from utils import show_images
from utils import train_step, test_step
from utils import get_cifar10_dataset, make_dataloader
from utils import simulate_scheduler
from utils import BaselineModel

from collections import  OrderedDict as odict

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ##### layers here ####
        self.convs = nn.Sequential(odict([
          ('conv1', nn.Conv2d(3, 64, 3, 1, 1)),
          ('relu1', nn.ReLU(inplace=True)),

          ('conv2', nn.Conv2d(64, 128, 3, 2, 1)),
          ('relu2', nn.ReLU(inplace=True)),
          ('conv3', nn.Conv2d(128, 128, 3, 1, 1)),
          ('relu3', nn.ReLU(inplace=True)),

          ('conv4', nn.Conv2d(128, 256, 3, 2, 1)),
          ('relu4', nn.ReLU(inplace=True)),
          ('conv5', nn.Conv2d(256, 256, 3, 1, 1)),
          ('relu5', nn.ReLU(inplace=True)),

          ('conv6', nn.Conv2d(256, 512, 3, 2, 1)),
          ('relu6', nn.ReLU(inplace=True)),
          ('conv7', nn.Conv2d(512, 512, 3, 1, 1)),
          ('relu7', nn.ReLU(inplace=True)),
        ]))
    
        self.fcs = nn.Sequential(odict([
          ('fc1', nn.Linear(512 * 4 * 4, 512)),
          ('fc2', nn.Linear(512, 256)),
          ('fc3', nn.Linear(256, 10)),
        ]))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
              nn.init.zeros_(m.bias)


    def forward(self, x):
        #### convolutions ####
        x = self.convs(x)
        ######################
        x = x.flatten(1)
        ####      FCs     ####
        x = self.fcs(x)
        ######################
        return x

print(MyCNN())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
momentum = 0.9
phases = ['train', 'test']

num_epochs = 50
learning_rate = 0.05
batch_size = 128

phases = ['train', 'test']
transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
cifar10_dataset = {
    phase: D.CIFAR10(
        'data', train=phase=='train', transform=transform, download=True
    )
    for phase in phases
}
loader = {
    phase: DataLoader(
        cifar10_dataset[phase],
        batch_size=batch_size,
        shuffle=phase=='train'
    )
    for phase in ['train', 'test']
}
images, target = sample_random_data(cifar10_dataset['train'])
titles = [cifar10_dataset['train'].classes[idx] for idx in target]
show_images(images.permute(0,2,3,1), titles)

data_augmentation = True

dataset = get_cifar10_dataset(random_crop=data_augmentation)
loader = make_dataloader(dataset, batch_size)

gamma = 0.85
lrs = simulate_scheduler(gamma, num_epochs)
plt.plot(lrs)
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.show()

model = MyCNN().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

pp = ProgressPlot(
    plot_names=phases,
    line_names=['loss', 'accuracy'],
    x_lim=[0, None],
    x_label='Iteration',
    y_lim=[[0, None], [0, 100]]
)

accuracy = 0
for epoch in range(num_epochs):
    for inputs, target in loader['train']:
        loss = train_step(model, inputs, target, optimizer, criterion, device)
        pp.update([[loss, -1], [-500, accuracy]])
    
    corrects = 0
    for inputs, target in loader['test']:
        output, _ = test_step(model, inputs, target, device=device)
        corrects += (output.argmax(1).cpu() == target).sum().item()
    accuracy = corrects / len(dataset['test']) * 100
    
    print(f'Epoch: {epoch+1} accuracy {accuracy:.2f}')
pp.finalize()

