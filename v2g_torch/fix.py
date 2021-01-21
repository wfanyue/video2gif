import os

import cv2
import utils

video_path='data/32.mp4'
video_name=os.path.splitext(os.path.split(video_path)[1])[0]
print(video_name)

# video = VideoFileClip(video_path)

videoCap = cv2.VideoCapture(video_path)

fps = videoCap.get(cv2.CAP_PROP_FPS)

frames = []
while videoCap.isOpened():
    _, frame = videoCap.read()
    if frame is None:
        break
    frames.append(frame)
videoCap.release()

# Build segments (uniformly of 5 seconds)
segments = [(start, int(start+fps*5)) for start in range(0, len(frames), int(fps*5))]

scores = utils.get_scores_use_opencv(segments, frames, stride=8)
print(scores)
