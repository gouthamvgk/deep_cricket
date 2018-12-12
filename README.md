# Classification of cricket video using Neural Network

## Overview

A LSTM based sequence model is created to classify every ball of a cricket match.  The model showed better result when trained with soft attention mechanism.  The attention mechanism was similar to the one used in the paper [Action recognition using visual attention](https://arxiv.org/pdf/1511.04119.pdf). 

## Dependencies

 - Python 3.6
 - Pytorch
 - Pandas
 - Opencv
 - ffmpeg
 - Moviepy

## Installation

All the dependency packages can be installed with `pip install` command.
If anaconda distribution is installed then `conda install` can be used.

## Data
Data is the most important thing for training neural network. Here I have collected highlights videos of IPL 2018 matches.  Highlights of 9 matches was used for training and 3 matches for validation.  Labelling the task was done manually.  In every video the start and end time of every delivery and the shot type was labelled and collected in .csv file. For every delivery in a video 5 labels were obtained,<br>
**Start time** - Start time of the ball<br>
**End time** - End time of the ball<br>
**Batsman type** - Right or left handed batsman<br>
**Bowler type** - Right or left handed bowler<br>
**Shot type** - Four, Six or Wicket.<br>

## Pre processing and Augmentation

From the highlights each delivery was trimmed out and then frames was obtained from each trimmed video which is the input to the network.<br>
**Pre processing** - From the full length video each delivery is trimmed out and then specified number of frames at regular interval is obtained for each ball.<br>
**Augmentation** - Since the dataset constructed by above method is every small we carry out random sampling of frames from the video.<br>
**Sampling** - The total frames in the video is divided into specified number of windows and then from each window a frame is sampled randomly.<br>


## Architecture
Resnet-50 is used as the CNN and LSTM is the RNN unit. Since for every delivery 8 frames(my config) are present a pretrained resnet model was used to obtain the  feature map for each frame.  Out of the 8 timesteps , the output of the final 3 time steps are passed through a fully connected layer to obtain the classification scores for each label.

Since attention mechanism is used instead of obtaining the feature vector from the resnet, an intermediate feature map is obtained and each position in the feature map is weighted with a softmax score and then given as an input to the LSTM(detailed formula in paper cited above).
So at each time step the model learns to concentrate only on certain regions which helps in accurate classification. 

## Training
For running the model from terminal use`python train_and_validate.py`

The model was trained with and without using attention. The model with attention showed better results compared to the other. Since it is a small dataset it was easily overfitting the training data and performed poorly on the validation set. So L2 regulariztion was included in the loss to prevent this.

## Performance
The model was trained for 20 epochs after which the loss stopped improving.  Among all the three labels Bowler type is the one which had least accuracy of 63% in test set. Batsman type and Shot type had accuracy of 79% and 89% respectively.

The reduced accuracy of bowler and batsman type is due to the fact that both the players occur maximum in two frames at the starting time steps. This makes it difficult to classify them. And in the shot type most of the errors were on the wickets.  This is because of the sparsity of the wicket class in the dataset and also the irregularities as a wicket can happen in any method like caught, bowled, stumping or run out.

The reduced performance is also due to irregularity in the camera angle for each ball. Since it is a highlights video,  the camera angle from which the shot is captured varies for each delivery. This makes it difficult for the model to capture and label the distribution.

## Issues
The dataset used for training was manually constructed by obtaining the full length videos and then trimming out each ball from it.  The time interval was annotated manually so while trimming there are some instances where frames containing vital information like batsman and bowler are distorted. So this also plays a role in the decreased accuracy of the classification scores.  

## Results


## Follow-up work
A proper dataset can be obtained where large amount of data is present with camera angles being constant and if each delivery is provided with a sentence describing it, then it can be used along with NLP methods for automatic commentary of each ball.

## References
The following papers were also referred:

 - [Show, Attend and Tell.](https://arxiv.org/pdf/1502.03044.pdf)
 - [LRCN for visual recognition and description.](https://arxiv.org/pdf/1411.4389.pdf)
 
