import os
from moviepy.editor import *

video_raw_path = "video_raw_fps/"
video_new_path = "video_30_fps/"

video_list = os.listdir(video_raw_path)
video_list = sorted(video_list, key=lambda x: int(str(x).split('.')[0]))
for i in video_list:
    clip = VideoFileClip(video_raw_path + i)
    print(clip.fps, clip.duration)
    clip.write_videofile(video_new_path + i, fps=25)
# clip = VideoFileClip(video_input_path)
# clip.write_videofile(video_output_path, fps=30)
# clip.reader.close()
