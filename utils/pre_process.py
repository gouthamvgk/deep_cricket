import cv2
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import random
from math import *
from video_to_frames import *


def video_to_frame(path, data, resize = False, size = None, times = 1, save_p = "/home/"):


    """
    This function does the pre processing required for sampling and
    frame capturing and then calls the two functions for video to
    frame conversion
    path: path to the video file_path
    data:A list containing information about given video
    resize: If yes then image resized
    size: Gives the resizing measure
    times: Tells how much time a video has to sampled.  This acts as data augmentation
           as a video is sampled randomly every time.
    """


    name = os.path.basename(path).split('.mp4')[0] #extracting the video name for creating folder for each data
    csv = []
    for i in range(times):
        save_path = save_p + name + '_' + str(i+1)
        csv.append([name + '_' + str(i+1)] + data)
        os.mkdir(save_path)
        sample = sampling(path)
        if resize:
            FrameCapture(path, sample, save_path, size = size)
        else:
            FrameCapture(path, sample, save_path)
    return csv



def pre_processing(path, limit = 4, video_path, target_path):
    """
    This function splits each delviery video into frames
    path: path to the csv file that contains information about all videos_path
    limit: If the video is >= limit then it is sample mulitple times randomly
    """
    label = []
    obj = pd.read_csv(path)
    for i in range(len(obj)):
        file_path = video_path + obj['File'][i]
        if(obj['Time'][i] < limit):
            da = video_to_frame(file_path, data = [obj['Batsman'][i], obj['Bowler'][i], obj['Delivery'][i]], resize = True, size = (224,224))
        elif(obj['Time'][i] >= limit):
            if(obj['Delivery'][i] == 1 or obj['Delivery'][i] == 2):
                da = video_to_frame(file_path, data = [obj['Batsman'][i], obj['Bowler'][i], obj['Delivery'][i]],resize = True, size = (224,224), times = 2)
            elif(obj['Delivery'][i] == 3):
                da = video_to_frame(file_path, data = [obj['Batsman'][i], obj['Bowler'][i], obj['Delivery'][i]], resize = True, size = (224,224),times = 3)
        label += da
    csv = pd.DataFrame(label, columns = ['File', 'Batsman', 'Bowler', 'Delivery'])
    csv.to_csv(target_path +  '/processed_label_2.csv', index = False)
