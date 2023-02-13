# video2gif

A demo using PyTorch simple implementation of [Video2gif](https://arxiv.org/abs/1605.04850) for video highlight selection.

### How to Use(only for academic purpose)
1. To process a demo video:
put your video path such as "test.mp4" in line 14 in try.py: 
set the time of highlight clip in line 15 in seconds, (e.g.,  5 for 5 seconds)
```
video_path = 'test.mp4'
highlight_time = 5 # seconds
```
run 
```
python try.py
```

2. For batch process
put your videos under the flod such as "video_25_fps" in line 14 in batch_process.py:
```
video_path='video_25_fps/'
```
run
```
python test_fps25.py
```

### Reference & Acknowledgement
Gygli, Michael, Yale Song, and Liangliang Cao. "Video2gif: Automatic generation of animated gifs from video." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.


