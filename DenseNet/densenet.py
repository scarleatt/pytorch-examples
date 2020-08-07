import torch as torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np
import os

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        # we let each 1×1 convolution 
        # produce 4k feature-maps.
        inner_channel = 4*growth_rate
        
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)
    
    
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
        
    def forward(self, x):
        return self.down_sample(x)
    
    
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=10):
        super().__init__()
        self.growth_rate = growth_rate
        
        """
        Before entering the ﬁrst dense block, a convolution with 16 
        (or twice the growth rate for DenseNet-BC) output channels is 
        performed on the input images.
        """
        inner_channels = 2*growth_rate
        
        """
        For convolutional layers with kernel size 3×3, each side of 
        the inputs is zero-padded by one pixel to keep the feature-map 
        size ﬁxed.
        """
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)
        
        self.features = nn.Sequential()
        
        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index),
                                    self._make_dense_layer(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate*nblocks[index]
            
            #"""generate θm output feature maps"""
            out_channels = int(reduction*inner_channels)
            self.features.add_module("transition_layer_{}".format(index),
                                    Transition(inner_channels, out_channels))
            inner_channels = out_channels
            
        # last dense layer
        self.features.add_module("dense_block{}".format(len(nblocks)-1),
                                self._make_dense_layer(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate*nblocks[len(nblocks)-1]
        
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        
        self.linear = nn.Linear(inner_channels, num_class)
        
        
    def forward(self, x):
        o = self.conv1(x)
        o = self.features(o)
        o = self.avgpool(o)
        o = o.view(o.size()[0], -1)
        o = self.linear(o)
        return o
        
        
    def _make_dense_layer(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                           download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                             shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                     (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_densenet.pth'
    torch.save(model.state_dict(), PATH)
    print("Model saved in ./cifar_densenet.pth !")
    
    img = Image.open("./images/cat.jpg")

    trans_ops = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    images = trans_ops(img).view(-1, 3, 224, 224)
    print(images)
    outputs = model(images)

    _, predictions = outputs.topk(5, dim=1)
    print('------------------------------------')
    print('open ./images/cat.jpg')

    for idx in predictions.numpy()[0]:
        print("Predicted labels:", classes[idx])