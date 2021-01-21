import list2tensor
import numpy as np

import torch
import torch.nn as nn
import cv2

c3d_weight_file = "data/c3d_model.pkl"
video2gif_weight_file = "data/video2gif_weights.npz"


class Video2GifNet(nn.Module):

    def __init__(self):
        super(Video2GifNet, self).__init__()
        # ----------- 1st layer group ---------------
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3),
                                        padding=(1, 1, 1)).double()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)).double()

        # ----------- 2nd layer group ---------------
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)).double()

        # ----------- 3rt layer group ---------------
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double()
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)).double()

        # ----------- 4th layer group ---------------
        self.conv4a = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double()
        self.conv4b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double()
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)).double()

        # ----------- 5th layer group ---------------
        self.conv5a = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double()
        self.conv5b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1)).double()

        self.pad = nn.ConstantPad3d(padding=(0, 1, 0, 1, 0, 0), value=0)
        #  (padding_left, padding_right. padding_top, padding_bottom,
        #   padding_front, padding_back )
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)).double()

        self.fc6 = nn.Linear(in_features=8192, out_features=4096).double()  # 5th layer out: 512 * input_depth = 16
        self.h1 = nn.Linear(in_features=4096, out_features=512).double()
        self.h2 = nn.Linear(in_features=512, out_features=128).double()

        self.score = nn.Linear(in_features=128, out_features=1).double()

        self.relu = nn.ReLU()

        self.__load_pretrained_weights()

    def forward(self, input_x):

        h = self.relu(self.conv1(input_x))

        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        # t = h
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))

        h = self.pad(h)

        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.relu(self.h1(h))
        h = self.relu(self.h2(h))

        score = self.score(h)

        return score

    def __load_pretrained_weights(self):
        # Initialize network.
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "fc.1.weight": "fc6.weight",
            "fc.1.bias": "fc6.bias",
            # h1
            "hLayer.1.weight": "h1.weight",
            "hLayer.1.bias": "h1.bias",
            # h2
            "hLayer.2.weight": "h2.weight",
            "hLayer.2.bias": "h2.bias",
            # score
            "score.weight": "score.weight",
            "score.bias": "score.bias",
        }

        p_dict = list2tensor.get_model_parameters_of_torch()
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)


def get_snips_opencv(images, image_mean, start=0, with_mirrored=False):
    # assert len(images) >= start + 16, "Not enough frames to fill a snipplet of 16 frames"
    # Convert images to caffe format and stack them
    assert len(images) >= start + 16, "Not enough frames to fill a snipplet of 16 frames"

    # Convert images to caffe format and stack them
    caffe_imgs = map(lambda x: bgr2caffe(x).reshape(1, 3, 128, 171), images[start:start + 16])
    snip = np.vstack(caffe_imgs).swapaxes(0, 1)

    # Remove the mean
    snip -= image_mean

    # Get the center crop
    snip = snip[:, :, 8:120, 29:141]
    snip = snip.reshape(1, 3, 16, 112, 112)

    if with_mirrored:  # Return normal and flipped version
        return np.vstack((snip, snip[:, :, :, :, ::-1]))
    else:
        return snip


def bgr2caffe(im, out_size=(128, 171), copy=True):
    # if copy:
    #     im = np.copy(im)
    # if len(im.shape) == 2:  # Make sure the image has 3 channels
    #     im = color.gray2rgb(im)

    h, w, _ = im.shape
    im = cv2.resize(im, out_size)
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    return np.array(im, dtype=np.float64)


class FlipFilterNet(nn.Module):

    def __init__(self):
        super(FlipFilterNet, self).__init__()
        # ----------- 1st layer group ---------------
        self.conv1 = nn.ConvTranspose3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3),
                                        padding=(1, 1, 1)).double()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)).double()

        # ----------- 2nd layer group ---------------
        self.conv2 = nn.ConvTranspose3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3),
                                        padding=(1, 1, 1)).double()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)).double()

        # ----------- 3rt layer group ---------------
        self.conv3a = nn.ConvTranspose3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1)).double()
        self.conv3b = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1)).double()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)).double()

        # ----------- 4th layer group ---------------
        self.conv4a = nn.ConvTranspose3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1)).double()
        self.conv4b = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1)).double()
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)).double()

        # ----------- 5th layer group ---------------
        self.conv5a = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1)).double()
        self.conv5b = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1)).double()

        self.pad = nn.ConstantPad3d(padding=(0, 1, 0, 1, 0, 0), value=0)
        #  (padding_left, padding_right. padding_top, padding_bottom,
        #   padding_front, padding_back )
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)).double()

        self.fc6 = nn.Linear(in_features=8192, out_features=4096).double()  # 5th layer out: 512 * input_depth = 16
        self.h1 = nn.Linear(in_features=4096, out_features=512).double()
        self.h2 = nn.Linear(in_features=512, out_features=128).double()

        self.score = nn.Linear(in_features=128, out_features=1).double()

        self.relu = nn.ReLU()


    def forward(self, input_x):
        h = self.relu(self.conv1(input_x))

        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        # t = h
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))

        h = self.pad(h)

        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.relu(self.h1(h))
        h = self.relu(self.h2(h))

        score = self.score(h)

        return score
