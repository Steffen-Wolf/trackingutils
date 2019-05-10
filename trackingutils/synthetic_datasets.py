from torch.utils.data import Dataset, DataLoader
from inferno.io.core import Zip, ZipReject
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.image import RandomFlip, RandomCrop
from inferno.io.transform.generic import Normalize
from inferno.io.transform.image import RandomRotate, ElasticTransform
# from neurofire.transform.affinities import Segmentation2Affinities2D, Segmentation2Affinities
from inferno.io.volumetric import LazyHDF5VolumeLoader, HDF5VolumeLoader, LazyN5VolumeLoader
# from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel
# from neurofire.criteria.loss_transforms import InvertTarget
from embeddingutils.transforms import Segmentation2AffinitiesWithPadding
from pathlib import Path
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
                 image_shape=(256, 256), digit_shape=(28, 28)):
        self.epoch_length = fake_dataset_size
        self.seq_len = seq_len
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
        self.mnist = load_dataset()
        moving_objects["sprite_images"] = [self.mnist[r]
                                           for r in np.random.randint(0, self.mnist.shape[0], self.nums_per_image)]

    def render_frame(self, frame_array, moving_objects):
        for i, digit in enumerate(moving_objects["sprite_images"]):
            x, y = int(moving_objects["positions"][i][0]), int(moving_objects["positions"][i][1])
            print(frame_array.shape, x, x + self.digit_shape[0], y, y + self.digit_shape[1])
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

        data = np.zeros((self.seq_len, self.img_shape[0], self.img_shape[1]))

        for frame_idx in range(self.seq_len):
            self.render_frame(data[frame_idx], moving_objects)
            self.update_positions(moving_objects)
            self.update_sprites(moving_objects)

        return np.clip(data, 0, 1)

    def __len__(self):
        return self.epoch_length


class MovingShapes(MovingMnist):

    def get_random_shape(self, noise_stength=3., mean_intensiy=100):
        import PIL.ImageDraw as ImageDraw
        import PIL.Image as Image

        image = Image.new("L", self.digit_shape)

        draw = ImageDraw.Draw(image)

        # define the base polygon
        max_x, max_y = self.digit_shape
        points = np.array([[max_x // 2, max_y // 8],
                           [max_y // 2 - max_y // 16, max_y // 2],
                           [max_x // 2, max_y - max_y // 8],
                           [max_x // 2 + max_y // 16, max_y // 2]],
                          dtype=np.float32)
        # shift to com
        points[:, 0] -= max_x // 2
        points[:, 1] -= max_y // 2

        # add random perturbations
        points += noise_stength * np.random.rand(*points.shape)

        # rotate by a random angle
        theta = 2 * np.random.rand() * np.pi
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
        points = rotMatrix.dot(points.T).T

        # shift back to center of image
        points[:, 0] += max_x // 2
        points[:, 1] += max_y // 2

        draw.polygon(tuple(map(tuple, points)),
                     fill=int(np.random.normal(loc=mean_intensiy, scale=5.0)))

        return np.array(image) / 255.

    def initialize_sprites(self, moving_objects):
        moving_objects["sprite_images"] = [self.get_random_shape()
                                           for i in range(self.nums_per_image)]


class MovingShapesUnsupervised(MovingMnist):
    def __getitem__(self, index):
        out = super().__getitem__(index).astype(np.float32)
        return out[None], np.stack(out, (out > 0).astype(np.float32))

if __name__ == '__main__':
    from ctctracking.loss import UnsupervisedLoss

    ms = MovingShapes(image_shape=(256, 256), digit_shape=(128, 128))

    model_parameters = {}
    model_parameters["self_mean"] = [0, 0, 0, 0, 0, 0, 0, 0]
    model_parameters["self_std"] = [0, 0, 0, 0, 0, 0, 0, 0]
    model_parameters["self_keys"] = ["size", "avg_int", "moment_of_inertia", "maxdist", "Q_00", "Q_01", "Q_11"]
    model_parameters["self_cov"] = [[0]]

    model_parameters["pair_mean"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    model_parameters["pair_std"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    model_parameters["pair_keys"] = ["size", "avg_int", "com_1",
                                     "com_0", "moment_of_inertia", "maxdist", "Q_00", "Q_01", "Q_11"]
    model_parameters["pair_cov"] = [[0]]

    usl = UnsupervisedLoss("", dim=2,
                           fov=[64, 64],
                           n_samples=20,
                           scale=.01,
                           model_parameters=model_parameters)

    feat = []

    for k in range(100):
        image_window = ms.get_random_shape()
        p_incluster = image_window > 0
        print(p_incluster.sum())
        feat.append(usl.computed_self_features(torch.from_numpy(image_window.astype(np.float32)),
                                               torch.from_numpy(p_incluster.astype(np.float32))))

    feat = np.stack(feat)
    print(feat.shape)

    print(",".join([str(x) for x in np.mean(feat, axis=0)]))
    print(",".join([str(x) for x in np.std(feat, axis=0)]))
