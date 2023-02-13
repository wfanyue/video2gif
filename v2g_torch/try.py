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


video_path = 'test.mp4'
highlight_time = 5
stride = 8


start = time.time()
print("start to score the video")
video = VideoFileClip(video_path)
segments = [(start, int(start+video.fps*highlight_time)) for start in range(0, int(video.duration*video.fps), int(video.fps*highlight_time))]
scores = utils.get_scores(segments, video, stride=stride)
id_of_highlight = utils.get_highlight_idx(scores)
print("score the video ", id_of_highlight, "takes ", time.time() - start, "s")
