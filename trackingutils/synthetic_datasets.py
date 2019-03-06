from torch.utils.data import Dataset, DataLoader
from inferno.io.core import Zip, ZipReject
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.image import RandomFlip, RandomCrop
from inferno.io.transform.generic import Normalize
from inferno.io.transform.image import RandomRotate, ElasticTransform
from neurofire.transform.affinities import Segmentation2Affinities2D, Segmentation2Affinities
from inferno.io.volumetric import LazyHDF5VolumeLoader, HDF5VolumeLoader, LazyN5VolumeLoader
from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel
from neurofire.criteria.loss_transforms import InvertTarget
from embeddingutils.transforms import Segmentation2AffinitiesWithPadding
from pathlib import Path
from .transforms import SliceTransform
import h5py
import torch
import numpy as np
import vigra

import sys
import os
import math


###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# adopted from Tencia Lee
###########################################################################################

# helper functions


def arr_from_img(im, shift=0):
    w, h = im.size
    arr = im.getdata()
    c = np.product(arr.size) / (w * h)
    return np.asarray(arr, dtype=np.float32).reshape((h, w, c)).transpose(2, 1, 0) / 255. - shift


def get_picture_array(X, index, shift=0):
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = ((X[index] + shift) * 255.).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)
    return load_mnist_images('train-images-idx3-ubyte.gz')


class MovingMnist(Dataset):

    def __init__(self, seq_len=20, nums_per_image=2, fake_dataset_size=1000,
                 image_shape=(64, 64), digit_shape=(28, 28)):
        self.epoch_length = fake_dataset_size
        self.seq_len = seq_len
        self.mnist = load_dataset()
        self.nums_per_image = nums_per_image
        self.img_shape = image_shape
        self.digit_shape = digit_shape
        self.x_lim = self.img_shape[0] - self.digit_shape[0]
        self.y_lim = self.img_shape[1] - self.digit_shape[1]
        self.lims = (self.x_lim, self.y_lim)

    def initialize_positions(self, moving_objects):
        moving_objects["positions"] = [(np.random.rand() * self.x_lim, np.random.rand()
                                        * self.y_lim) for _ in range(self.nums_per_image)]

    def initialize_movement(self, moving_objects):
        directions = np.pi * (np.random.rand(self.nums_per_image) * 2 - 1)
        speeds = np.random.randint(5, size=self.nums_per_image) + 2
        moving_objects["veloc"] = [(v * math.cos(d), v * math.sin(d)) for d, v in zip(directions,
                                                                                      speeds)]

    def initialize_sprites(self, moving_objects):
        moving_objects["mnist_images"] = [self.mnist[r]
                                          for r in np.random.randint(0, self.mnist.shape[0], self.nums_per_image)]

    def render_frame(self, frame_array, moving_objects):
        for i, digit in enumerate(moving_objects["mnist_images"]):
            x, y = int(moving_objects["positions"][i][0]), int(moving_objects["positions"][i][1])
            frame_array[x:x + self.digit_shape[0], y:y + self.digit_shape[1]] += digit[0]

    def update_positions(self, moving_objects):
        # update positions based on velocity
        next_pos = [list(map(sum, zip(p, v))) for p, v in zip(moving_objects["positions"],
                                                              moving_objects["veloc"])]
        # bounce off wall if a we hit one
        for i, pos in enumerate(next_pos):
            for j, coord in enumerate(pos):
                if coord < 0 or coord > self.lims[j]:
                    moving_objects["veloc"][i] = tuple(
                        list(moving_objects["veloc"][i][:j]) +
                        [-1 * moving_objects["veloc"][i][j]] +
                        list(moving_objects["veloc"][i][j + 1:]))

        moving_objects["positions"] = [list(map(sum, zip(p, v)))
                                       for p, v in zip(moving_objects["positions"],
                                                       moving_objects["veloc"])]

    def update_sprites(self, moving_objects):
        pass                  

    def __getitem__(self, index):
        moving_objects = {}
        self.initialize_positions(moving_objects)
        self.initialize_movement(moving_objects)
        self.initialize_sprites(moving_objects)

        self.get_moving_sprites()

        data = np.zeros((self.seq_len, 64, 64))

        for frame_idx in range(self.seq_len):
            self.render_frame(data[frame_idx], moving_objects)
            self.update_positions(moving_objects)
            self.update_sprites(moving_objects)

        return np.clip(data, 0, 1)

    def __len__(self):
        return self.epoch_length


class MovingShapes(MovingMnist):

    def initialize_sprites(self, moving_objects):
        moving_objects["mnist_images"] = [self.mnist[r]
                                          for r in np.random.randint(0, self.mnist.shape[0], self.nums_per_image)]


if __name__ == '__main__':
    pass
