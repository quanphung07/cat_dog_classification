import os
import torch 
import torchvision
import torch.nn as nn
from torchvision import models,transforms

import matplotlib.pyplot as plt
transform_umnorm=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])
def make_model():
    model=models.mobilenet_v2()
    model.classifier[1]=nn.Linear(in_features=1280,out_features=2)
    model.load_state_dict(torch.load('model_weights.pth',map_location=torch.device('cpu')))
    return model
class BaseTransform():
    def __init__(self,size,mean,std):
        self.transform={'train': transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
]),
                       'val': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])}
    def __call__(self,img,phase='train'):
        return self.transform[phase](img)
