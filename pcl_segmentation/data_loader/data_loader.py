# ==============================================================================
# MIT License
#
# Copyright 2021 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import os
import random
import numpy as np
import scipy.ndimage
import tensorflow as tf


class DataLoaderSeq(tf.keras.utils.Sequence):
  def __init__(self, image_set, data_path, mc, use_fraction=None):
    self.mc = mc

    validation = True if image_set == "val" else False
    self._batch_size = mc.BATCH_SIZE if not validation else 1
    self._data_augmentation = mc.DATA_AUGMENTATION if not validation else False

    self._image_set = image_set
    self._data_root_path = data_path
    self._lidar_2d_path = os.path.join(self._data_root_path, image_set)

    self._image_idx = self._load_image_set_idx()

    if use_fraction is not None and use_fraction != 1.0:
      print("Use only {0} training samples".format(use_fraction))
      self._image_idx = self._image_idx[: int(len(self._image_idx) * use_fraction)]

  @property
  def image_idx(self):
    return self._image_idx

  @property
  def image_set(self):
    return self._image_set

  @property
  def data_root_path(self):
    return self._data_root_path

  def _load_image_set_idx(self):
    image_set_file = os.path.join(
      self._data_root_path, 'ImageSet', self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]

    # shuffle files
    random.shuffle(image_idx)

    return image_idx

  def _lidar_2d_path_at(self, idx):
    lidar_2d_path = os.path.join(self._lidar_2d_path, idx + '.npy')

    assert os.path.exists(lidar_2d_path), \
      'File does not exist: {}'.format(lidar_2d_path)
    return lidar_2d_path

  def __len__(self):
    return len(self._image_idx) // self._batch_size

  def __getitem__(self, idx):
    mc = self.mc
    batch_idx = self._image_idx[idx * self._batch_size:(idx + 1) * self._batch_size]

    lidar_per_batch = []
    lidar_mask_per_batch = []
    label_per_batch = []
    weight_per_batch = []

    for idx in batch_idx:
      # load data
      record = np.load(self._lidar_2d_path_at(idx)).astype(np.float32, copy=False)
      INPUT_MEAN = mc.INPUT_MEAN
      INPUT_STD = mc.INPUT_STD
      if self._data_augmentation:
        if mc.RANDOM_FLIPPING:
          if np.random.rand() > 0.5:
            # flip y
            record = record[:, ::-1, :]
            record[:, :, 1] *= -1
            INPUT_MEAN[:, :, 1] *= -1
        if mc.RANDOM_SHIFT:
          random_y_shift = np.random.random_integers(low=-75, high=75)
          record = scipy.ndimage.shift(input=record,
                                       shift=[0, random_y_shift, 0],
                                       order=0,
                                       mode='constant',
                                       cval=0.0)

      lidar = record[:, :, :5]  # x, y, z, intensity, depth
      lidar_mask = np.reshape(  # binary mask
        (lidar[:, :, 4] > 0),
        [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
      )
      # normalize
      lidar = (lidar - INPUT_MEAN) / INPUT_STD

      # set input on all channels to zero where no points are present
      lidar[~np.squeeze(lidar_mask)] = 0.0

      # append mask to lidar input
      lidar = np.append(lidar, lidar_mask, axis=2)

      # get label label from record
      label = record[:, :, 5]

      # set label to None class where no points are present
      label[~np.squeeze(lidar_mask)] = mc.CLASSES.index("None")

      weight = np.zeros(label.shape)
      for l in range(mc.NUM_CLASS):
        weight[label == l] = mc.CLS_LOSS_WEIGHT[int(l)]

      # Append all the data
      lidar_per_batch.append(lidar)
      label_per_batch.append(label)
      lidar_mask_per_batch.append(lidar_mask)
      weight_per_batch.append(weight)

    lidar_per_batch = np.array(lidar_per_batch)
    label_per_batch = np.array(label_per_batch)
    lidar_mask_per_batch = np.array(lidar_mask_per_batch)
    weight_per_batch = np.array(weight_per_batch)

    lidar_per_batch = lidar_per_batch.astype('float32')
    label_per_batch = label_per_batch.astype('int32')
    lidar_mask_per_batch = lidar_mask_per_batch.astype('bool')
    weight_per_batch = weight_per_batch.astype('float32')

    return (lidar_per_batch, lidar_mask_per_batch), label_per_batch, weight_per_batch
