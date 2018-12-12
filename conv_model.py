import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from __future__ import print_function, division



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Choosing the device makes the tensor computation to run on GPU when it
is available.
"""



class image_encoder(nn.Module):


    """
    This module is used to extract the convolution feature map of
    the input image.
    feature_size: The output feature size we want.
    finetune_last: The number of layers from last that we want to finetune
                  during training
    """



    def __init__(self, feature_size = 7, finetune_last = 1):
        super(image_encoder, self).__init__()
        layer1 = models.resnet50(pretrained = True)
        self.layer = nn.Sequential(*list(layer1.children())[:-2]) #taking only upto the conv layers in resnet
        del(layer1)
        if(finetune_last != None):
            self.finetune(finetune_last)
        self.output = nn.AdaptiveAvgPool2d((feature_size, feature_size)) #Input image of any size is made to the required feature_size dimension



    def forward(self, images):
        feature_map = self.layer(images)
        correct_map = self.output(feature_map)
        return correct_map


co
    def finetune(self, last_layer):
        for k in list(self.layer.children())[:-last_layer]:
            for params in k.parameters():
                params.requires_grad = False  #making requires_grad = False for layers that we dont want to train
