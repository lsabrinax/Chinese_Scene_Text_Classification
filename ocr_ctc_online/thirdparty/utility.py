"""Contains common utility functions."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import distutils.util
import numpy as np
from paddle.fluid import core
import paddle.fluid as fluid
import six
import cv2
import random

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int32")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = core.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def get_ctc_feeder_data(data, place, need_label=True):
    pixel_tensor = core.LoDTensor()
    pixel_data = None
    pixel_data = np.concatenate(
        list(map(lambda x: x[0][np.newaxis, :], data)), axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    label_tensor = to_lodtensor(list(map(lambda x: x[1], data)), place)
    if need_label:
        return {"pixel": pixel_tensor, "label": label_tensor}
    else:
        return {"pixel": pixel_tensor}


def get_ctc_feeder_for_infer(data, place):
    return get_ctc_feeder_data(data, place, need_label=False)


def get_attention_feeder_data(data, place, need_label=True):
    #print data.shape
    #print ("data shape")
    #print (data)
    #print ("data")
    pixel_tensor = core.LoDTensor()
    pixel_data = None
    pixel_data = np.concatenate(
        list(map(lambda x: x[0][np.newaxis, :], data)), axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    label_in_tensor = to_lodtensor(list(map(lambda x: x[1], data)), place)
    label_out_tensor = to_lodtensor(list(map(lambda x: x[2], data)), place)
    if need_label:
        return {
            "pixel": pixel_tensor,
            "label_in": label_in_tensor,
            "label_out": label_out_tensor
        }
    else:
        return {"pixel": pixel_tensor}


def get_attention_feeder_for_infer(data, place):
    batch_size = len(data)
    init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_recursive_seq_lens = [1] * batch_size
    init_recursive_seq_lens = [init_recursive_seq_lens, init_recursive_seq_lens]
    init_ids = fluid.create_lod_tensor(init_ids_data, init_recursive_seq_lens,
                                       place)
    init_scores = fluid.create_lod_tensor(init_scores_data,
                                          init_recursive_seq_lens, place)

    pixel_tensor = core.LoDTensor()
    pixel_data = None
    pixel_data = np.concatenate(
        list(map(lambda x: x[0][np.newaxis, :], data)), axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    return {
        "pixel": pixel_tensor,
        "init_ids": init_ids,
        "init_scores": init_scores
    }

class ImageTransfer(object):
    """crop, add noise, change contrast, color jittering"""
    def __init__(self, image):
        """image: a ndarray with size [h, w, 3]"""
        self.image = image

    def slight_crop(self):
        h, w = self.image.shape[:2]
        k = random.random() * 0.08  # 0.0 <= k <= 0.1
        ch, cw = int(h * 0.9), int(w - k * h)     # cropped h and w
        hs = random.randint(0, h - ch)      # started loc
        ws = random.randint(0, w - cw)
        return self.image[hs:hs+ch, ws:ws+cw]

    def add_noise(self):
        img = self.image * (np.random.rand(*self.image.shape) * 0.2 + 0.8)
        img = img.astype(np.uint8)
        return img

    def change_contrast(self):
        # if random.random() < 0.5:
        #     k = random.randint(7, 9) / 10.0
        # else:
        #     k = random.randint(11, 13) / 10.0
        # b = 128 * (k - 1)
        # img = self.image.astype(np.float)
        # img = k * img - b
        # img = np.maximum(img, 0)
        # img = np.minimum(img, 255)
        # img = img.astype(np.uint8)
        r = np.random.RandomState(0)
        clahe = cv2.createCLAHE(clipLimit=round(r.uniform(0, 2), 2), tileGridSize=(8,8))
        b, g, r = cv2.split(self.image)
        b1 = clahe.apply(b)
        g1 = clahe.apply(g)
        r1 = clahe.apply(r)
        img = cv2.merge((b1, g1, r1))
        return img

    def perspective_transform(self):
        h, w = self.image.shape[:2]
        short = min(h, w)
        gate = int(short * 0.1)
        mrg = []
        for _ in range(8):
            mrg.append(random.randint(0, gate))
        pts1 = np.float32(
            [[mrg[0], mrg[1]], [w - 1 - mrg[2], mrg[3]], [mrg[4], h - 1 - mrg[5]], [w - 1 - mrg[6], h - 1 - mrg[7]]])
        pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, M, (w, h))

    def gamma_transform(self, a=1.0, gamma=2.0):
        image = self.image.astype(np.float)
        image = image / 255
        image = a * (image ** gamma)
        image = image * 255
        image = np.minimum(image, 255)
        image = image.astype(np.uint8)
        return image

    def change_hsv(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        s = random.random()
        def ch_h():
            dh = random.randint(2, 11) * random.randrange(-1, 2, 2)
            img[:, :, 0] = (img[:, :, 0] + dh) % 180
        def ch_s():
            ds = random.random() * 0.25 + 0.7
            img[:, :, 1] = ds * img[:, :, 1]
        def ch_v():
            dv = random.random() * 0.4 + 0.6
            img[:, :, 2] = dv * img[:, :, 2]
        if s < 0.25:
            ch_h()
        elif s < 0.50:
            ch_s()
        elif s < 0.75:
            ch_v()
        else:
            ch_h()
            ch_s()
            ch_v()
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def random_augmentation(image, allow_crop=True):
    f = ImageTransfer(image)
    seed = random.randint(0, 6)     # 0: original image used
    switcher = random.random() if allow_crop else 1.0
    if seed == 1:
        image = f.add_noise()
    elif seed == 2:
        image = f.change_contrast()
    elif seed == 3:
        image = f.change_hsv()
    elif seed == 4:
        a = random.random() * 0.4 + 0.8
        gamma = random.random()
        image = f.gamma_transform(a=a, gamma=gamma)
    elif seed >= 5:
        f1 = ImageTransfer(f.add_noise())
        f2 = ImageTransfer(f1.change_hsv())
        f3 = ImageTransfer(f2.gamma_transform(1.0, 1.5))
        image = f3.change_contrast()
    if switcher < 0.4:
        fn = ImageTransfer(image)
        image = fn.slight_crop()
    elif switcher < 0.7:
        fn = ImageTransfer(image)
        image = fn.perspective_transform()
    return image