import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from predict import cricket
import glob
import cv2
import numpy as np
from __future__ import print_function, division
import pandas as pd
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train_and_test(batch_details, model ,transforms,video_path, criterion = None, optimizer = None,  sample_rate = 8, phase = 'train'):
    batch_size = len(batch_details)
    image = torch.zeros(batch_size*sample_rate, 3, 224, 224)
    image1 = torch.zeros(batch_size*sample_rate, 3, 224, 224)
    h_c = model.c_h_init(batch_size)
    h_c = h_c.to(device)
    files = []
    a,b,c = [],[],[]
    for folder in batch_details:
        if (phase == 'train'):
            order = glob.glob(video_path +"/train" + folder[0] + "/*.jpg")
            order.sort()
            files += order
        else:
            order = glob.glob(video_path +"/test" + folder[0] + "/*.jpg")
            order.sort()
            files += order
        a.append(folder[1]-1)
        b.append(folder[2]-1)
        c.append(folder[3]-1)
    for no, loc in enumerate(files):
        img = cv2.imread(loc)
        img = transforms(img)
        image[no] = img
    pos = 0
    for i in range(sample_rate):
        for j in range(batch_size):
            image1[pos] = image[i+(j*sample_rate)]
            pos += 1
    del(image)
    image1 = image1.to(device)
    a = torch.LongTensor(a)
    b = torch.LongTensor(b)
    c = torch.LongTensor(c)
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)
    if (phase == 'train'):
        model.train()
        predict = model(image1,h_c,h_c, batch_size)
        loss1 = criterion(predict[0],a)
        loss2 = criterion(predict[1],b )
        loss3 = criterion(predict[2],c)

        loss = loss3 + loss1 + loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.eval()
        predict2 = model(image1, h_c, h_c, batch_size)
    elif(phase == 'test'):
        model.eval()
        predict2 = model(image1, h_c, h_c, batch_size)

    _, pre_1 = torch.max(predict2[0], dim = 1)
    _, pre_2 = torch.max(predict2[1], dim = 1)
    _, pre_3 = torch.max(predict2[2], dim = 1)
    a1 = (sum(pre_1 == a).item())/batch_size
    a2 = (sum(pre_2 == b).item())/batch_size
    a3 = (sum(pre_3 == c).item())/batch_size
    if (phase == 'train'):
        return loss.item(), (a1,a2,a3)
    else:
        return (a1,a2,a3)



def tt_iterations(no_epochs, csv_train, model,optimizer, video_path, csv_test = None ,  batch_size = 5, sample_rate = 8, print_every = 10, epo_test = False, file = None, save_path, save_every):
    obj1 = pd.read_csv(csv_train)
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for k in range(no_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(k+1, no_epochs))
        print('Training...')
        obj1 = obj1.iloc[np.random.permutation(len(obj1))]
        su = []
        accu_1 = []
        accu_2 = []
        accu_3 = []
        batch = 1
        for j in range(0,len(obj1), batch_size):
            lower = j
            upper = j + (batch_size) if (j+batch_size) <= len(obj1)  else len(obj1)
            if (lower == upper+1):
                data = obj1.iloc[lower].values.tolist()
                data = [data]
            else:
                data = obj1.iloc[lower:upper].values.tolist()
            loss, (a1,a2,a3) = train_and_test(data,model, trans, criterion, optimizer,video_path, sample_rate, 'train')
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

        if(file):
            file.write('Epoch {}'.format(k+1))
            file.write('Epoch completed in {:.0f}minutes {:.0f}seconds\n'.format(elapsed_time//60, elapsed_time%60))
            file.write('Loss for {} epoch is {}\n'.format(k+1, sum(su)/len(su)))
            file.write('Training set accuracy:\n')
            file.write('Batsman accuracy-->{}\n'.format(sum(accu_1)/len(accu_1)))
            file.write('Bowler accuracy-->{}\n'.format(sum(accu_2)/len(accu_2)))
            file.write('Shot accuracy-->{}\n'.format(sum(accu_3)/len(accu_3)))

        print('---------------------------------------------------------------------------------')
        if (epo_test):
            test(csv_test, model,trans,video_path, batch_size, sample_rate , file)
            print('*********************************************************************************')
        if(save_path != None):
            if ((k+1) % save_every == 0):
                torch.save({'model': model.state_dict(),
                             'optimizer': optimizer.state_dict()}, save_path +'/model' + str(k+1), ".pth")




def test(csv_test, model, transforms,video_path, batch_size = 5, sample_rate = 8, file = None):
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
        (a1,a2,a3) = train_and_test(data,model, transforms,video_path, sample_rate = sample_rate, phase = 'test')
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
    model1 = cricket()
    model1 = model1.to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr = predictor_lr)
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
    save_path = "" #path to save the model

    tt_iterations(no_epochs, path1,model1, optimizer1, video_path, path2, batch_size, sample_rate,print_every, epo_test, f , save_path, save_every)
