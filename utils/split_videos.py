import cv2
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import random
from math import *
import glob

def split_videos(labels_path, videos_path, target_path):


    """
    This function splits the given videos into small chunks of videos
    corresponding to each delivery. It takes the csv files in the label path
    which has a separate file for each video and corresponding video file
    from videos_path and does the splitting according to the csv data"""


    no_videos = len(glob.glob(labels_path + "/*.csv"))
    
    for i in range(1,no_videos+1):
        file1 = labels_path + "/" + str(i) + ".csv" #csv file must be named like "1.csv", 2.csv
        file2 = videos_path + "/" + str(i) + ".mp4" #video file must be named like "1.mp4", "2.mp4"
        obj = pd.read_csv(file1)
        for j in range(len(obj)):
            target = target_path + "/" + str(count) + ".mp4"
            ffmpeg_extract_subclip(file2 , obj['s'][j], obj['e'][j], targetname=target)
            csv.append(["video" + str(count) + ".mp4", obj['ba'][j], obj['bo'][j], obj['ty'][j], i,j+1, obj['e'][j] - obj['s'][j] ])
            count = count + 1
    csv = pd.DataFrame(csv, columns = ['File', 'Batsman', 'Bowler', 'Delivery', 'Video', 'index', 'Time'])
    csv.to_csv(targets_path + "/processed_file.csv", index = False)
