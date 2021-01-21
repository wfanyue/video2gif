import os
import time

from moviepy.video.io.VideoFileClip import VideoFileClip
import model
import torch
import list2tensor
import warnings
import numpy as np
import utils
warnings.filterwarnings("ignore")


video_path = 'video_25_fps/'
video_list = os.listdir(video_path)
video_list = sorted(video_list, key=lambda x: int(x.split('.')[0]))
highlight_time = 5
stride = 8

f = open("result_25_fps_without_sampling.txt", 'a')
f.write("result of video with 25 fps, highlight time is " + str(highlight_time) + "s a segment sample only 16 frames " + " \n")
for i in video_list:
    start = time.time()
    print("start to score the video", i)
    video = VideoFileClip(video_path + i)
    segments = [(start, int(start+video.fps*highlight_time)) for start in range(0, int(video.duration*video.fps), int(video.fps*highlight_time))]
    scores = utils.get_scores(segments, video, stride=stride)
    print(scores)
    id_of_highlight = utils.get_highlight_idx(scores)
    print("score the video ", i, "takes ", time.time() - start, "s", "the duration is ", str(video.duration))
    text1 = "\nthe video of " + i.split('.')[0] + " takes " + str(time.time() - start) + "s " + "the duration is " + str(video.duration)
    f.write(text1)
    text2 = "\nthe top idx of segment " + str(id_of_highlight) + " is: "
    for key in scores.keys():
        temp = str(scores[key]).split(',')[0].split('(')[1]  # 分割出得分
        text2 += "[" + str(key) + " " + temp + "] "
    f.write(text2 + "\n")
