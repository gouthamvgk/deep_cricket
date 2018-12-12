import cv2
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import random
from math import *
import glob







def sampling(path, window_size = 8, diff_percent = 20):


    """
    This function is used to determine the frame numbers of frames to be sampled from all the frames
    of the given video.
    path: Specifies the path to the video file
    window_size: The no of frames that has to be sampled from the given video
    diff_percent: This tells by how much amount the choosen frames in successive frame should
                  lie from each other
    """

    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #determines the no. of frames in the video
    win = []
    ran = floor(length/window_size) #Determines the window size of the given video
    """
    Determines the upper and lower index of each window
    """
    for i in range(1, window_size+1):
        lower = (i-1) * ran + 1
        upper = i * ran
        win.append([lower, upper])
    win[-1][1] = length
    samp = []
    per = ceil((diff_percent/100) * ran)
    sa = random.randint(win[0][0], win[0][1])
    samp.append(sa)
    prev = sa
    for i in win[1:]:
        k = True
        while(k):
            sa = random.randint(i[0], i[1])
            if (sa-prev > per): #Checks whether the choosen frames between adjacent windows differ by given amount if not then radomly sampled again
                k = False
        samp.append(sa)
        prev = sa
    return samp




def FrameCapture(path, sample, save_path, size = None):
    """
    This function is used to capture the specified frames from the
    given video file.
    path: path to the video file
    sample: A list containing the frame numbers that are to be sample
    save_path: path to save the sampled images
    size: If specified then the frame will resized to the given size.
    """

    cap = cv2.VideoCapture(path)
    count = 1
    success = 1
    frame_no = 1
    while success:
        success, image = cap.read()
        if(sample[0] == count):
            image = image[0:350, :, :] #removes the bottom 50 pixels that contains the scorecard
            if (size):
                image = cv2.resize(image, size) #resizing the image if True
            cv2.imwrite(save_path + "/" + str(frame_no) + "frame" + str(count) + ".jpg", image)
            frame_no += 1
            sample = sample[1:]
            if (sample == []):
                break
        count += 1
