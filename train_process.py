import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import glob
import cv2
import numpy as np
from __future__ import print_function, division
import pandas as pd
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Choosing the device makes the tensor computation to run on GPU when it
is available.
"""





def train_and_test(batch_details, model1, model2 ,transforms,video_path, probs_c = 0.3, criterion = None, optimizer1 = None, optimizer2 = None, sample_rate = 8, phase = 'train'):


    """
    This function carries out the training process of the network for each step.  It validates on test set for each epoch on True.
    model1 : convolution network module
    model2 : Prediction network module
    transforms : A transform object that contains transforms for applying on input image
    prob_c : weight of the attention term that we use in loss computation.
    criterion : Loss object.
    optimizer1, optimizer2 : Objects for doing optimisation on convolution and prediction module.
    sample_rate : No of frames for each video which if of fixed length
    phase : Tells whether it is a training or testing phase
    """

    batch_size = len(batch_details)
    image = torch.zeros(batch_size*sample_rate, 3, 224, 224) #computes the image input for convolution network
    files = []
    a,b,c = [],[],[]
    for folder in batch_details:

        """
        The frames for each video are put in separate folder.
        So for every instance of the batch, collecting the respective
        file names of the frames, for image construction.
        """

        if (phase == 'train'):
            order = glob.glob(video_path + "/train" + folder[0] + "/*.jpg")
            order.sort()
            files += order
        else:
            order = glob.glob(video_path + "/test" + folder[0] + "/*.jpg")
            order.sort()
            files += order
        a.append(folder[1]-1)
        b.append(folder[2]-1)
        c.append(folder[3]-1)  #target tensors for each output
    for no, loc in enumerate(files):
        img = cv2.imread(loc)
        img = transforms(img)
        image[no] = img  #applying transforms to images
    image = image.to(device)


    a = torch.LongTensor(a) #criterion expects ground truth vectors in LongTensor format
    b = torch.LongTensor(b)
    c = torch.LongTensor(c)
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)


    if (phase == 'train'):
        model1.train()
        model2.train()
        compute_map = model2(image) #computing the feature map.
        predict, probs = model1(compute_map)
        loss1 = criterion(predict[0],a)
        loss2 = criterion(predict[1],b )
        loss3 = criterion(predict[2],c)
        probs = probs.permute(1,0,2)
        loss = loss3 + loss1 + loss2 + probs_c * (((1-probs.sum(dim=1))**2).sum(dim=1).mean()) #adding the stochastic loss of attention weights as mentioned in the paper.
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward() #backward propagation of the loss
        optimizer1.step()
        optimizer2.step()
        model1.eval()
        model2.eval()
        compute_map = model2(image)
        predict2, _ = model1(compute_map)
    elif(phase == 'test'):
        model1.eval()
        model2.eval()
        compute_map = model2(image)
        predict2, _ = model1(compute_map) #computing the prediction in test phase.


      """
      Finding the best class from the predicted outputs
      """
    _, pre_1 = torch.max(predict2[0], dim = 1)
    _, pre_2 = torch.max(predict2[1], dim = 1)
    _, pre_3 = torch.max(predict2[2], dim = 1)
      """
      Comparing the predicted and ground truth vectors.
      """
    a1 = (sum(pre_1 == a).item())/batch_size
    a2 = (sum(pre_2 == b).item())/batch_size
    a3 = (sum(pre_3 == c).item())/batch_size
    if (phase == 'train'):
        return loss.item(), (a1,a2,a3)
    else:
        return (a1,a2,a3)
