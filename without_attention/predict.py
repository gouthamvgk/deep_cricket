import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from __future__ import print_function, division
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class cricket(nn.Module):



    def __init__(self, sample_rate = 8, hidden_size = 500, input_size = 1000):
        super(cricket, self).__init__()
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layer = models.resnet50(pretrained = True)
        for params in self.layer.parameters():
            params.requires_grad = False
        self.layer.fc.weight.requires_grad = True
        self.layer.fc.bias.requires_grad = True
        self.lstm = nn.LSTM(self.input_size,self.hidden_size)
        self.out1 = nn.Linear(self.hidden_size, 2)
        self.out2 = nn.Linear(self.hidden_size, 2)
        self.out3 = nn.Linear(self.hidden_size,3)




    def forward(self, inp, hidden, cell, batch_size):


        features = self.layer(inp)
        hn, cn = (hidden, cell)
        for i in range(0,self.sample_rate - 1):
            index = i * batch_size
            if index not in [self.sample_rate-1, self.sample_rate-2, self.sample_rate-3]:
                _, (hn,cn) = self.lstm(features[index:index+batch_size].view(1, batch_size, self.input_size), (hn,cn))
            elif (index == 5):
                output1, (hn,cn) = self.lstm(features[index:index+batch_size].view(1,batch_size, self.input_size), (hn,cn))
            elif (index == 6):
                output2, (hn,cn) = self.lstm(features[index:index+batch_size].view(1,batch_size, self.input_size), (hn,cn))
            elif (index == 7):
                output3, (hn,cn) = self.lstm(features[index:index+batch_size].view(1,batch_size, self.input_size), (hn,cn))

        x = F.dropout(output1[0])
        y = F.dropout(output2[0])
        z = F.dropout(output3[0])
        x = self.out1(hn[0])
        y = self.out2(hn[0])
        z = self.out3(hn[0])

        x = F.softmax(x, dim = 1)
        y = F.softmax(y, dim = 1)
        z = F.softmax(z, dim = 1)
        return x,y,z



    def c_h_init(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
