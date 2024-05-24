# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2 as cv
import numpy as np
import torch


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)


def torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).numpy()


def torch_to_npimage(a: torch.Tensor, unnormalize=True, input_bgr=False):
    a_np = torch_to_numpy(a.clamp(0.0, 1.0))

    if unnormalize:
        a_np = a_np * 255
    a_np = a_np.astype(np.uint8)

    if input_bgr:
        return a_np

    return cv.cvtColor(a_np, cv.COLOR_RGB2BGR)
