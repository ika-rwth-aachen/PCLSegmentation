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

import tensorflow as tf


class PCLSegmentationNetwork(tf.keras.Model):
  """Base Class for segmentation Networks which implements the Loss funcion, train step and tensorboard methods"""

  def __init__(self, mc):
    super(PCLSegmentationNetwork, self).__init__()
    self.mc = mc
    self.NUM_CLASS = mc.NUM_CLASS
    self.BATCH_SIZE = mc.BATCH_SIZE
    self.ZENITH_LEVEL = mc.ZENITH_LEVEL
    self.AZIMUTH_LEVEL = mc.AZIMUTH_LEVEL
    self.NUM_FEATURES = mc.NUM_FEATURES

    self.FOCAL_GAMMA = mc.FOCAL_GAMMA
    self.CLS_LOSS_COEF = mc.CLS_LOSS_COEF
    self.DENOM_EPSILON = mc.DENOM_EPSILON

    self.CLASSES = mc.CLASSES
    self.CLS_COLOR_MAP = mc.CLS_COLOR_MAP

    self.softmax = tf.keras.layers.Softmax(axis=-1)

    # Metrics
    self.miou_tracker = tf.keras.metrics.MeanIoU(num_classes=self.NUM_CLASS, name="MeanIoU")
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")

  def call(self, inputs, training=False, mask=None):
    raise NotImplementedError("Method should be called in child class!")

  def segmentation_head(self, logits, lidar_mask):
    with tf.name_scope("segmentation_head") as scope:
      probabilities = self.softmax(logits)

      predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)

      # set predictions to the "None" class where no points are present
      predictions = tf.where(tf.squeeze(lidar_mask),
                             predictions,
                             tf.ones_like(predictions) * self.CLASSES.index("None")
                             )
    return probabilities, predictions

  def focal_loss(self, probabilities, lidar_mask, label, loss_weight):
    """Focal Loss"""
    with tf.name_scope("focal_loss") as scope:
      lidar_mask = tf.cast(lidar_mask, tf.float32)  # from bool to float32

      label = tf.reshape(label, (-1,))  # class labels

      prob = tf.reshape(probabilities, (-1, self.NUM_CLASS)) + self.DENOM_EPSILON  # output prob

      onehot_labels = tf.one_hot(label, depth=self.NUM_CLASS)  # onehot class labels

      cross_entropy = tf.multiply(onehot_labels, -tf.math.log(prob)) * \
                      tf.reshape(loss_weight, (-1, 1)) * tf.reshape(lidar_mask, (-1, 1))  # cross entropy

      weight = (1.0 - prob) ** self.FOCAL_GAMMA  # weight in the def of focal loss

      fl = weight * cross_entropy  # focal loss

      loss = tf.identity(tf.reduce_sum(fl) / tf.reduce_sum(lidar_mask) * self.CLS_LOSS_COEF, name='class_loss')

    return loss

  def train_step(self, data):
    (lidar_input, lidar_mask), label, loss_weight = data

    with tf.GradientTape() as tape:
      probabilities, predictions = self([lidar_input, lidar_mask], training=True)  # forward pass
      loss = self.focal_loss(probabilities, lidar_mask, label, loss_weight)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update & Compute Metrics
    with tf.name_scope("metrics") as scope:
      self.loss_tracker.update_state(loss)
      self.miou_tracker.update_state(label, predictions)
      loss_result = self.loss_tracker.result()
      miou_result = self.miou_tracker.result()
    return {'loss': loss_result, 'miou': miou_result}

  def test_step(self, data):
    (lidar_input, lidar_mask), label, loss_weight = data

    probabilities, predictions = self([lidar_input, lidar_mask], training=False)  # forward pass

    loss = self.focal_loss(probabilities, lidar_mask, label, loss_weight)

    # Update Metrics
    self.loss_tracker.update_state(loss)
    self.miou_tracker.update_state(label, predictions)

    return {'loss': self.loss_tracker.result(), 'miou': self.miou_tracker.result()}

  def predict_step(self, data):
    (lidar_input, lidar_mask), _, _ = data

    probabilities, predictions = self([lidar_input, lidar_mask], training=False)  # forward pass
    return probabilities, predictions

  @property
  def metrics(self):
    # We list the `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    return [self.loss_tracker, self.miou_tracker]

  def get_config(self):
    config = super(PCLSegmentationNetwork, self).get_config()
    config.update({"mc": self.mc})
    return config
