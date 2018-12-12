import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from train_process import train_and_test
from conv_model import image_encoder
from predict_model import label, att_net
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






def tt_iterations(no_epochs, csv_train, model1, model2,optimizer1,optimizer2,video_path,  csv_test = None ,  batch_size = 5, sample_rate = 8, print_every = 10,save_every = 5, epo_test = False, file = None,save_path=None):


    """
    This function performs the whole training process and reports the epoch statistics upon completion.
    no_epochs: Number of epochs the training has to be performed
    csv_train: path to csv file that contains information about training set(name, ground truth values)
    model1: The Label object for predictions.
    model2: The conv net object for computing feature map.
    optimizer1, optimizer2: The optimisation object for model1 and model2
    csv_test: path to csv file that contains informationn about test set. Not required if epo_test is False
    batch_size: Batch size of the training process.
    Sample_rate: No of frames for each video
    print_every: No. of batches after which running loss has to be printed in each epoch.
    epo_test: Indicates whether to validate test set after each epoch.
    file: File to which the training and testing statistics has to be logged.
    save_every: epochs after which models and optimizer has to be saved.
    """


    obj1 = pd.read_csv(csv_train)
    """
    Trans object composes various transforms to be performed on the input image
    ToTensor: converts the image to tensor and makes the (0-255) to (0-1) range
    Normalize: Normalizes the image tensor values based upon the predefined conv net.
    """


    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
    criterion = torch.nn.CrossEntropyLoss() #loss object to be passed for training
    optimizer1.zero_grad()
    optimizer2.zero_grad()



    for k in range(no_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(k+1, no_epochs))
        print('Training...')
        obj1 = obj1.iloc[np.random.permutation(len(obj1))] #permuting video files randomly on each epoch
        su = []
        accu_1 = [] #list to hold batsman accuracy
        accu_2 = [] #list to hold bowler accuracy
        accu_3 = [] #list to hold shot accuracy
        batch = 1
        for j in range(0,len(obj1), batch_size):
            lower = j
            upper = j + (batch_size) if (j+batch_size) <= len(obj1)  else len(obj1)
            if (lower == upper+1):
                data = obj1.iloc[lower].values.tolist()
                data = [data]
            else:
                data = obj1.iloc[lower:upper].values.tolist()
            loss, (a1,a2,a3) = train_and_test(data,model1,model2, trans,video_path, 0.3, criterion, optimizer1, optimizer2, sample_rate, 'train')
            accu_1.append(a1)
            accu_2.append(a2)
            accu_3.append(a3)
            su.append(loss)
            if((batch % print_every) == 0):
                print('Running loss after {} batches is {}'.format(batch, sum(su)/len(su)))
            batch += 1
        elapsed_time = time.time() - since



        print('Epoch completed in {:.0f}minutes {:.0f}seconds'.format(elapsed_time//60, elapsed_time%60))
        print('Loss for {} epoch is {}\n'.format(k+1, sum(su)/len(su)))
        print('Training set accuracy:')
        print('Batsman accuracy-->{}'.format(sum(accu_1)/len(accu_1)))
        print('Bowler accuracy-->{}'.format(sum(accu_2)/len(accu_2)))
        print('Shot accuracy-->{}'.format(sum(accu_3)/len(accu_3)))

        if(file): #if file is given logs the outputs to that file
            file.write('Epoch {}'.format(k+1))
            file.write('Epoch completed in {:.0f}minutes {:.0f}seconds\n'.format(elapsed_time//60, elapsed_time%60))
            file.write('Loss for {} epoch is {}\n'.format(k+1, sum(su)/len(su)))
            file.write('Training set accuracy:\n')
            file.write('Batsman accuracy-->{}\n'.format(sum(accu_1)/len(accu_1)))
            file.write('Bowler accuracy-->{}\n'.format(sum(accu_2)/len(accu_2)))
            file.write('Shot accuracy-->{}\n'.format(sum(accu_3)/len(accu_3)))

        print('---------------------------------------------------------------------------------')
        if (epo_test):
            test(csv_test, model1, model2 ,trans,video_path, batch_size, sample_rate , file)
            print('*********************************************************************************')
            """
            Saving the state of the models and optimizer after every 5 epochs for future validation.
            """
        if(save_path != None):
            if ((k+1) % save_every == 0):
                torch.save({
                        'model1':model1.state_dict(),
                        'model2':model2.state_dict(),
                        'opt1':optimizer1.state_dict(),
                        'opt2': optimizer2.state_dict()
                          }, save_path + '/att_' + str(k+1)+ '.pth')





def test(csv_test, model1, model2, transforms,video_path, batch_size=5 , sample_rate=8, file = None):


    """
    This function is used to validate the model on test set data.
    Works like train function except it works on the test
    """


    print('Validating...')
    obj2 = pd.read_csv(csv_test)
    accu_1 = []
    accu_2 = []
    accu_3 = []
    for j in range(0,len(obj2), batch_size):
        lower = j
        upper = j + (batch_size) if (j+batch_size) <= len(obj2)  else len(obj2)
        if (lower == upper+1):
            data = obj2.iloc[lower].values.tolist()
            data = [data]
        else:
            data = obj2.iloc[lower:upper].values.tolist()
        (a1,a2,a3) = train_and_test(data,model1, model2, transforms,video_path, sample_rate = sample_rate, phase = 'test')
        accu_1.append(a1)
        accu_2.append(a2)
        accu_3.append(a3)
    print('Test set accuracy:')
    print('Batsman accuracy-->{}'.format(sum(accu_1)/len(accu_1)))
    print('Bowler accuracy-->{}'.format(sum(accu_2)/len(accu_2)))
    print('Shot accuracy-->{}'.format(sum(accu_3)/len(accu_3)))

    if(file):
        file.write('Test set accuracy:\n')
        file.write('Batsman accuracy-->{}\n'.format(sum(accu_1)/len(accu_1)))
        file.write('Bowler accuracy-->{}\n'.format(sum(accu_2)/len(accu_2)))
        file.write('Shot accuracy-->{}\n'.format(sum(accu_3)/len(accu_3)))
        file.write('-------------------------------------------------------------------------\n')



if __name__ == "__main__":
    predictor_lr = 0.0001
    conv_lr = 0.0004
    model1 = label()
    model2 = image_encoder()
    model1 = model1.to(device)
    model2 = model2.to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr = predictor_lr)
    optimizer2 = optim.Adam(model2.parameters(), lr = conv_lr)
    video_path "temp/"
    path1 = "" #path to the train set csv file
    path2 = "" #path to the test set csv file
    path3 = "" #path to the log file
    f = file.open(path3, 'a')
    sample_rate = 8
    batch_size = 10
    no_epochs = 10
    print_every = 10
    save_every = 5
    epo_test = True
    save_path = "" path to save the model

    tt_iterations(no_epochs, path1,model1, model2, optimizer1, optimizer2,video_path, path2, batch_size, sample_rate,print_every, save_every, epo_test, f , save_path)
