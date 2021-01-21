import pickle
import collections

import numpy as np
import torch

c3d_weight_file = "data/c3d_model.pkl"
c3d_weight_file_torch_pth = "data/c3d-pretrained.pth"
video2gif_weight_file = "data/video2gif_weights.npz"
print("Load pretrained c3d weights from %s", c3d_weight_file)
with open(c3d_weight_file, "rb") as f:
    m_c3d_weights = pickle.load(f, encoding="latin1")

print('Load pretrained autogif_demo weights from %s...' % video2gif_weight_file)
v2g_weights = np.load(video2gif_weight_file, allow_pickle=True, encoding='bytes')['arr_0']
v2g_model_list = m_c3d_weights[0:-4]
v2g_model_list.extend(list(v2g_weights))


def get_model_parameters_of_torch():
    c3d_weight_torch = torch.load(c3d_weight_file_torch_pth)
    v2g_model_torch = collections.OrderedDict()
    cnt = 0
    for key in c3d_weight_torch.keys():
        if cnt >= 16:
            break
        temp_tensor = torch.from_numpy(v2g_model_list[cnt]).double()    # 原模型float64
        v2g_model_torch[key] = temp_tensor
        cnt += 1

    temp_tensor = torch.from_numpy(v2g_model_list[16]).double()
    v2g_model_torch['fc.1.weight'] = temp_tensor.t()        # pytorch linear的特点，需要转置
    temp_tensor = torch.from_numpy(v2g_model_list[17]).double()
    v2g_model_torch['fc.1.bias'] = temp_tensor

    temp_tensor = torch.from_numpy(v2g_model_list[18]).double()
    v2g_model_torch['hLayer.1.weight'] = temp_tensor.t()
    temp_tensor = torch.from_numpy(v2g_model_list[19]).double()
    v2g_model_torch['hLayer.1.bias'] = temp_tensor

    temp_tensor = torch.from_numpy(v2g_model_list[20]).double()
    v2g_model_torch['hLayer.2.weight'] = temp_tensor.t()
    temp_tensor = torch.from_numpy(v2g_model_list[21]).double()
    v2g_model_torch['hLayer.2.bias'] = temp_tensor

    temp_tensor = torch.from_numpy(v2g_model_list[22]).double()
    v2g_model_torch['score.weight'] = temp_tensor.t()
    temp_tensor = torch.from_numpy(v2g_model_list[23]).double()
    v2g_model_torch['score.bias'] = temp_tensor
    return v2g_model_torch
