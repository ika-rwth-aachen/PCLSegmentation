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
import tensorflow as tf


def _tensor_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))


class DataLoader:
  def __init__(self, dataset_split, dataset_root_path, mc):
    """
    Arguments:
      dataset_split -- String containing "train", "val" or "test"
      dataset_root_path -- String containing path to the root directory of the splits
    """
    self.mc = mc

    validation = True if dataset_split == "val" else False
    self._batch_size = mc.BATCH_SIZE if not validation else 1
    self._data_augmentation = mc.DATA_AUGMENTATION if not validation else False

    self._dataset_split = dataset_split
    self._dataset_root_path = dataset_root_path
    self._dataset_path = os.path.join(self._dataset_root_path, dataset_split)
    self._sample_pathes = tf.io.gfile.glob(os.path.join(self._dataset_path, "*.npy"))

  @staticmethod
  def random_y_flip(lidar, mask, label, weight, prob=0.5):
    """
    Randomly flips a Numpy ndArray along the second axis (y-axis) with probability prob

    Arguments:
      lidar -- Numpy ndArray of shape [height, width, channels]
      mask -- Numpy ndArray of shape [height, width]
      label -- Numpy ndArray of shape [height, width]
      weight -- Numpy ndArray of shape [height, width]
      prob -- Float which describes the probability that the flip is applied on the sample.

    Returns:
      lidar -- Numpy ndArray of shape [height, width, channels]
      mask -- Numpy ndArray of shape [height, width]
      label -- Numpy ndArray of shape [height, width]
      weight -- Numpy ndArray of shape [height, width]
    """
    # creates a random float between 0 and 1
    random_float = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float16)

    (lidar, mask, label, weight) = tf.cond(
                             pred=random_float > prob,
                             true_fn=lambda: (tf.reverse(lidar, axis=[1]),
                                              tf.reverse(mask, axis=[1]),
                                              tf.reverse(label, axis=[1]),
                                              tf.reverse(weight, axis=[1])
                                              ),
                             false_fn=lambda: (lidar, mask, label, weight)
                             )

    return lidar, mask, label, weight

  @staticmethod
  def random_shift_lr(lidar, mask, label, weight, shift):
    """
    Randomly shifts a sample left-right

    Arguments:
      lidar -- Numpy ndArray of shape [height, width, channels]
      mask -- Numpy ndArray of shape [height, width]
      label -- Numpy ndArray of shape [height, width]
      weight -- Numpy ndArray of shape [height, width]
      shift -- Integer which defines the maximal amount of the random horizontal shift

    Returns :
      sample -- Numpy ndArray of shape [height, width, channels]
    """
    # Generate a random integer between in [-shift, shift]
    random = tf.random.uniform(shape=[], minval=-shift, maxval=shift, dtype=tf.int32)

    lidar = tf.roll(lidar, random, axis=1)
    mask = tf.roll(mask, random, axis=1)
    label = tf.roll(label, random, axis=1)
    weight = tf.roll(weight, random, axis=1)

    return lidar, mask, label, weight


  @staticmethod
  def random_shift_up_down(lidar, mask, label, weight, shift):
    """
    Randomly shifts a sample up-down

    Arguments:
      lidar -- Numpy ndArray of shape [height, width, channels]
      mask -- Numpy ndArray of shape [height, width]
      label -- Numpy ndArray of shape [height, width]
      weight -- Numpy ndArray of shape [height, width]
      shift -- Integer which defines the maximal amount of the random horizontal shift

    Returns :
      sample -- Numpy ndArray of shape [height, width, channels]
    """
    # Generate a random integer between in [-shift, shift]
    random = tf.random.uniform(shape=[], minval=-shift, maxval=shift, dtype=tf.int32)

    lidar = tf.roll(lidar, random, axis=0)
    mask = tf.roll(mask, random, axis=0)
    label = tf.roll(label, random, axis=0)
    weight = tf.roll(weight, random, axis=0)

    return lidar, mask, label, weight

  def parse_sample(self, sample_path):
    """
    Parses a data sample from a file path an returns a lidar tensor, a mask tensor and a label tensor

    Arguments:
      sample_path -- String - File path to a sample *.npy file

    Returns:
      lidar -- numpy ndarray of shape [height, width, num_channels] containing the lidar data
      mask -- numpy ndarray of shape  [height, width] containing a boolean mask
      label -- numpy ndarray of shape [height, width] containing the label as segmentation map
      weight -- numpy ndarray of shape [height, width] containing the weighting for each class
    """

    # Load numpy sample
    sample = np.load(sample_path.numpy()).astype(np.float32, copy=False)

    # Get x, y, z, intensity, depth
    lidar = sample[:, :, :5]

    # Compute binary mask: True where the depth is bigger then 0, false in any other case
    mask = lidar[:, :, 4] > 0

    # Normalize input data using the mean and standard deviation
    lidar = (lidar - self.mc.INPUT_MEAN) / self.mc.INPUT_STD

    # Set lidar on all channels to zero where the mask is False. Ie. where no points are present
    lidar[~mask] = 0.0

    # Add Dimension to mask to obtain a tensor of shape [height, width, 1]
    mask = np.expand_dims(mask, -1)

    # Append mask to lidar input
    lidar = np.append(lidar, mask, axis=2)

    # Squeeze mask
    mask = np.squeeze(mask)

    # Get segmentation map from sample
    label = sample[:, :, 5]

    # set label to None class where no points are present
    label[~mask] = self.mc.CLASSES.index("None")

    # construct class-wise weighting defined in the configuration
    weight = np.zeros(label.shape)
    for l in range(self.mc.NUM_CLASS):
      weight[label == l] = self.mc.CLS_LOSS_WEIGHT[int(l)]

    return lidar.astype('float32'), mask.astype('bool'), label.astype('int32'), weight.astype('float32')

  @staticmethod
  def serialize_sample(lidar, mask, label, weight):
    """
    Creates a tf.train.Example message ready to be written to a file.
    Arguments:
      lidar -- numpy ndarray of shape [height, width, num_channels] containing the lidar data
      mask -- numpy ndarray of shape  [height, width] containing a boolean mask
      label -- numpy ndarray of shape [height, width] containing the label as segmentation map
      weight -- numpy ndarray of shape [height, width] containing the weighting for each class

    Returns:
      example_proto -- Serialized train sample as String
    """
    feature = {
        'lidar': _tensor_feature(lidar),
        'mask': _tensor_feature(mask),
        'label': _tensor_feature(label),
        'weight': _tensor_feature(weight)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

  def tf_serialize_example(self, lidar, mask, label, weight):
    """
    Wrapper function for self.serialize_sample

    Arguments:
      lidar -- numpy ndarray of shape [height, width, num_channels] containing the lidar data
      mask -- numpy ndarray of shape  [height, width] containing a boolean mask
      label -- numpy ndarray of shape [height, width] containing the label as segmentation map
      weight -- numpy ndarray of shape [height, width] containing the weighting for each class

    Returns:
      example_proto -- Serialized train sample as String
    """
    tf_string = tf.py_function(
      self.serialize_sample,
      (lidar, mask, label, weight),   # Pass these args to the above function.
      tf.string)                      # The return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar.

  def serialize_dataset(self, sample_pathes):
    """
    Arguments:
      sample_pathes -- List of Strings which contain pathes for the training samples

    Returns:
      dataset -- tf.data.Dataset with serialized data
    """
    random.shuffle(sample_pathes)

    # create a tf.data.Dataset using sample_pathes
    dataset = tf.data.Dataset.from_tensor_slices(sample_pathes)

    # Apply parse_sample and read the *.npy file
    dataset = dataset.map(lambda sample:
                          tf.py_function(self.parse_sample, [sample], [tf.float32, tf.bool, tf.int32, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Apply self.tf_serialize_example on the parsed sample
    dataset = dataset.map(self.tf_serialize_example, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

  def write_tfrecord_dataset(self):
    """
    Write the TFRecord file for the current dataset. The TFRecord file is automatically written to
    self._dataset_root_path with the name train.tfrecord or val.tfrecord or test.tfrecord. If the file already exists
    this processing part is skipped.
    """
    filename = os.path.join(self._dataset_root_path, self._dataset_split + ".tfrecord")

    if os.path.isfile(filename):
      print("TFRecord exists at {0}. Skipping TFRecord writing.".format(filename))
    else:
      print("Writing TFRecord to {0}".format(filename))
      serialized_dataset = self.serialize_dataset(self._sample_pathes)
      writer = tf.data.experimental.TFRecordWriter(filename)
      writer.write(serialized_dataset)
    return self

  def parse_proto(self, example_proto):
    """
    Deserializes the sample proto String encoded with self.serialize_sample back to tf.Tensors.
    Also applies augmentation functions to the sample defined in the config.

    Arguments:
      example_proto -- Sample serialized as proto String
    Returns:
      lidar -- numpy ndarray of shape [height, width, num_channels] containing the lidar data
      mask -- numpy ndarray of shape  [height, width] containing a boolean mask
      label -- numpy ndarray of shape [height, width] containing the label as segmentation map
      weight -- numpy ndarray of shape [height, width] containing the weighting for each class
    """

    feature_description = {
      'lidar': tf.io.FixedLenFeature([], tf.string),
      'mask': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.string),
      'weight': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)

    lidar = tf.io.parse_tensor(example["lidar"], out_type=tf.float32)
    mask = tf.io.parse_tensor(example["mask"], out_type=tf.bool)
    label = tf.io.parse_tensor(example["label"], out_type=tf.int32)
    weight = tf.io.parse_tensor(example["weight"], out_type=tf.float32)

    lidar = tf.reshape(lidar, shape=[self.mc.ZENITH_LEVEL, self.mc.AZIMUTH_LEVEL, self.mc.NUM_FEATURES])
    mask = tf.reshape(mask, shape=[self.mc.ZENITH_LEVEL, self.mc.AZIMUTH_LEVEL])
    label = tf.reshape(label, shape=[self.mc.ZENITH_LEVEL, self.mc.AZIMUTH_LEVEL])
    weight = tf.reshape(weight, shape=[self.mc.ZENITH_LEVEL, self.mc.AZIMUTH_LEVEL])

    if self._data_augmentation:
      # Perform the random left-right flip augmentation
      if self.mc.RANDOM_FLIPPING:
        lidar, mask, label, weight = self.random_y_flip(lidar, mask, label, weight)

      # Perform the random left-right shift augmentation
      if self.mc.SHIFT_LEFT_RIGHT > 0:
        lidar, mask, label, weight = self.random_shift_lr(lidar, mask, label, weight, self.mc.SHIFT_LEFT_RIGHT)

      # Perform the random up-down shift augmentation
      if self.mc.SHIFT_UP_DOWN > 0:
        lidar, mask, label, weight = self.random_shift_up_down(lidar, mask, label, weight, self.mc.SHIFT_UP_DOWN)

    return (lidar, mask), label, weight

  def read_tfrecord_dataset(self, buffer_size=1000):
    """
    Arguments:
      buffer_size -- Shuffle buffer size
    Returns:
      dataset -- tf.data.Dataset containing the dataset
    """
    filename = os.path.join(self._dataset_root_path, self._dataset_split + ".tfrecord")
    raw_dataset = tf.data.TFRecordDataset([filename], num_parallel_reads=tf.data.AUTOTUNE)
    dataset = raw_dataset.map(self.parse_proto, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(self._batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
