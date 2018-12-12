import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from __future__ import print_function, division


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Choosing the device makes the tensor computation to run on GPU when it
is available.
"""




class att_net(nn.Module):


    """
    This module implements the attention network for determining region of interest in input image
    feature_size:no_of channels in convolution netowork output
    hidden_size: Hidden state dimension for the LSTM
    att_size: Dimension of the attention linear layers
    """


    def __init__(self, feature_size, hidden_size, att_size):
        super(att_net, self).__init__()
        self.f_to_a = nn.Linear(feature_size, att_size) #mapping from feature map to attention map
        self.h_to_a = nn.Linear(hidden_size, att_size) #mapping from hidden vector to attention map
        self.co_to_a = nn.Linear(att_size, 1) #mapping from combined hidden and feature attention map to attention weights
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) #makes the attention weights sum to 1




    def forward(self, encoded_image, hidd_vect):
        out_1 = self.f_to_a(encoded_image)
        out_2 = self.h_to_a(hidd_vect.squeeze(0))
        x = self.relu(out_1 + out_2.unsqueeze(1))
        combined = self.co_to_a(x).squeeze(2)
        prob = self.softmax(combined) #attention weights for each pixel
        weighted_att_map = (encoded_image * prob.unsqueeze(2)).sum(dim = 1) #feature map weighted by attention weights
        return prob, weighted_att_map







class label(nn.Module):


    """
    This module consists of the LSTM and implements the whole architecure
    feature_size:no_of channels in convolution netowork output
    hidden_size: Hidden state dimension for the LSTM
    sample_rate: No of frames sampled from each video which is of fixed value
    """


    def __init__(self, feature_size = 2048, hidden_size = 500, sample_rate = 8, att_size = 500):
        super(label, self).__init__()
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.att_net = att_net(feature_size, hidden_size, att_size) #initialising the attention network
        self.att_net = self.att_net.to(device)
        self.h_init = nn.Linear(feature_size, hidden_size) #Linear layer for hidden_state initialisation
        self.c_init = nn.Linear(feature_size, hidden_size) #Linear layer for cell_state initialisation
        self.lstm = nn.LSTM(feature_size,hidden_size)
        self.out1 = nn.Linear(hidden_size, 2) #Linear layer to classify batsman
        self.out2 = nn.Linear(hidden_size, 2) #Linear layer to classify Bowler
        self.out3 = nn.Linear(hidden_size,3) #Linear layer to classify shot
        self.init_weights()



    def init_weights(self, method = 'uniform'):


        """
        This function initialises the weight of the linear layers according to various types
        of initialisation methods proposed.
        """


        if (method == 'uniform'):
            self.out1.weight.data.uniform_(-0.1,0.1)
            self.out1.bias.data.fill_(0)
            self.out2.weight.data.uniform_(-0.1, 0.1)
            self.out2.bias.data.fill_(0)
            self.out3.weight.data.uniform_(-0.1,0.1)
            self.out3.bias.data.fill_(0)

        elif(method == 'xavier'):
            self.out1.weight.data.xavier_uniform_()
            self.out1.bias.data.fill_(0)
            self.out2.weight.data.xavier_uniform_()
            self.out2.bias.data.fill_(0)
            self.out3.weight.data.xavier_uniform_()
            self.out3.bias.data.fill_(0)

        elif(method == 'kaiming'):
            self.out1.weight.data.kaiming_uniform_()
            self.out1.bias.data.fill_(0)
            self.out2.weight.data.kaiming_uniform_()
            self.out2.bias.data.fill_(0)
            self.out3.weight.data.kaiming_uniform_()
            self.out3.bias.data.fill_(0)

    def init_h_c(self, feature_map, batch_size):


        """
        This method find the initial hidden and cell state vector.
        Since we are using attention which depeds on previous hidden state
        for current time step we initialise the hidden and cell state according to the
        feature map of inputs.
        """


        hc_init_temp = torch.zeros(batch_size,self.feature_size)
        hc_init_temp = hc_init_temp.to(device)
        for i in range(batch_size):
            lower = i * self.sample_rate
            upper = lower + self.sample_rate -1
            hc_init_temp[i] = (feature_map[lower:upper].sum(dim=1).sum(dim=0))/(self.sample_rate * feature_map.size(1)) #Summing over the dimensions of feature map
        initial_h = self.h_init(hc_init_temp)
        initial_c = self.c_init(hc_init_temp)
        initial_h = initial_h.unsqueeze(0)
        initial_c = initial_c.unsqueeze(0)

        return (initial_h, initial_c)

    def forward(self, compute_map):

        feature_map = compute_map.permute(0,2,3,1)
        total_map = feature_map.size(0)
        batch_size = int(feature_map.size(0) / self.sample_rate)
        feature_map = feature_map.view(total_map, -1, self.feature_size)  #changine the feature map size from so that all pixel positions are are made into one dimension.
        no_pixels = feature_map.size(1)
        hn, cn = self.init_h_c(feature_map, batch_size)
        attention_weights = torch.zeros(self.sample_rate, batch_size,  no_pixels)
        attention_weights = attention_weights.to(device) #in pytorch every initialisation made with torch.zeros goes into cpu, so changing it to gpu if necessary
        li1 = list()
        li2 = list()

        for time in range(self.sample_rate):
            for k in range(batch_size):
                li1.append(time+(k*self.sample_rate))
            li2.append(li1) #feature map consists of features of all batch in order, so extracting features required for every timestep
            li1 = []

        for time in range(self.sample_rate):
            prob, weighted_map = self.att_net(feature_map[li2[time]], hn) #feature map corresponding to each time step
            attention_weights[time] = prob #saving the attention weights for loss computation
            if(time not in [self.sample_rate-1, self.sample_rate-2, self.sample_rate-3]):
                _, (hn,cn) = self.lstm(weighted_map.unsqueeze(0), (hn, cn))

            """
            The output of initial timesteps are not required, so taking computation from lstm
            differently for different time steps.
            """
            
            elif(time == self.sample_rate -3):
                output, (hn,cn) = self.lstm(weighted_map.unsqueeze(0), (hn,cn))
                x = self.out1(output[0])
                x = F.softmax(x, dim=1) # classifier for batsman
            elif(time==self.sample_rate-2):
                output, (hn,cn) = self.lstm(weighted_map.unsqueeze(0), (hn,cn))
                y = self.out2(output[0])
                y = F.softmax(y, dim=1) #classifier for bowler
            elif(time==self.sample_rate-1):
                output, (hn,cn) = self.lstm(weighted_map.unsqueeze(0), (hn,cn))
                z = self.out3(output[0])
                z = F.softmax(z, dim=1) #classifier for shot

        return (x,y,z) , attention_weights
