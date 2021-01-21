import shutil

import moviepy
import os

video_save_dir = "video_raw_fps/"
video_file_dir = "/disk5/wfanyue-workdir/Projects/Video_Highlight_Annotation/dataset/"
video_all_list = os.listdir(video_file_dir)
# 而sorted()是python内置函数，不仅可以用于list，还可以用于其他类型数据。它不改变原有对象的值，而是新建一个列表，返回的是排好序的列表

# 按字符串排序文件名
v = sorted(video_all_list, key=lambda x: int(x.split('.')[0]))

for i in v:

    if 31 < int(i.split('.')[0]) < 60:
        print("copy file", i, "to ", video_save_dir)
        shutil.copyfile(video_file_dir + str(i), video_save_dir + str(i))

print(v)
