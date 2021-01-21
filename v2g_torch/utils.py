import collections
import queue
import threading
import time
import numpy as np
import torch
import model
import cv2

snippet_mean = np.load("data/snipplet_mean.npy")
video2gif_weight_file = "data/video2gif_weights.npz"
# Build model
net = model.Video2GifNet().cuda()


def get_scores_use_opencv(segments, video_frames, stride=8):
    map_of_segments = collections.OrderedDict()
    segments2score = collections.OrderedDict()
    seg_nr = 0
    frames = []

    for frame_idx, f in enumerate(video_frames):

        if frame_idx > segments[seg_nr][1]:
            seg_nr += 1  # 下一个片段
            if seg_nr == len(segments):
                break
            frames = []

        frames.append(f)    # 拾取帧

        if len(frames) == 16:
            snip = model.get_snips_opencv(
                images=frames,
                image_mean=snippet_mean,
                start=0,
                with_mirrored=True
            )
            score = predict(snip)
            # return score, t
            frames = frames[stride:]
            score_snip = score.mean()  # 一个batch取平均
            print(segments[seg_nr], score)

            if segments[seg_nr] not in map_of_segments:
                map_of_segments[segments[seg_nr]] = []  # 创建新的片段的得分列表
            map_of_segments[segments[seg_nr]].append(score_snip)

    for key in map_of_segments.keys():
        segments2score[key] = torch.tensor(0.0).cuda()
        for item in map_of_segments[key]:
            segments2score[key] += item
        segments2score[key] /= len(map_of_segments[key])

    return segments2score


def get_scores(segments, video, stride=8):
    map_of_segments = collections.OrderedDict()
    segments2score = collections.OrderedDict()

    frames = []
    seg_nr = 0
    for frame_idx, f in enumerate(video.iter_frames()):

        if frame_idx > segments[seg_nr][1]:
            seg_nr += 1
            if seg_nr == len(segments):
                break
            frames = []

        frames.append(f)

        if len(frames) == 16:  # Extract scores
            snip = model.get_snips(frames, snippet_mean, 0, False)
            t, m = predict(snip)
            score_snip = t.mean()  # 一个batch取平均
            return t, m
    #         frames = frames[stride:]  # shift by 'stride' frames
    #         if segments[seg_nr] not in map_of_segments:
    #             map_of_segments[segments[seg_nr]] = []  # 创建新的片段的得分列表
    #         print("score in an inference: ", score_snip)
    #         map_of_segments[segments[seg_nr]].append(score_snip)
    #
    # for key in map_of_segments.keys():
    #     print(len(map_of_segments[key]))
    #     segments2score[key] = torch.tensor(0.0).cuda()
    #     for item in map_of_segments[key]:
    #         segments2score[key] += item
    #     segments2score[key] /= len(map_of_segments[key])
    #
    # return segments2score


# 该函数用于预测分数, 模型做inference
def predict(clip_video):
    """
    Get prediction function
    @return: theano function that scores sniplets
    :param clip_video:
    """
    print('to inference......')
    clip_video = torch.from_numpy(clip_video).double()

    model_input_tensor = clip_video.cuda()

    with torch.no_grad():
        model_output = net.forward(model_input_tensor)  # inference

    return model_output


def get_highlight_idx(scores_of_segment):  # 取出得分最高的片段作为最精彩部分
    num_arr = np.zeros(shape=[len(scores_of_segment), 1])
    for key_index, key in enumerate(scores_of_segment.keys()):
        num_arr[key_index] = scores_of_segment[key].cpu().numpy()  # 转存入array中，用于求最大值
    return np.nanargmax(num_arr)


def store_highlight(video, video_name, index_of_segment, segments, path):
    start_frame = segments[index_of_segment][0]
    end_frame = segments[index_of_segment][1]

    start_time = start_frame / video.fps
    end_time = end_frame / video.fps
    time_of_highlight = end_time - start_time
    video_save = video.subclip(start_time, end_time)
    video_save.write_videofile(
        path + "/video_save_" + video_name + "_highlight_" + str(int(time_of_highlight)) + "s.mp4")
