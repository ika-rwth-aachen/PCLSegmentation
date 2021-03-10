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

import io
import itertools

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  # normalize confusion matrix
  cm_normalized = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Normalized Confusion Matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm_normalized[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure


def confusion_matrix_to_iou_recall_precision(cm):
  """
  Computes the classwise iou, recall and precision from a confusion matrix
  cm: Confusion matrix as nxn matrix where n is the number of classes
  Confusion matrix has switched axes when taken from *total_cm* !!
  """
  with tf.name_scope("compute_iou_recall_precision") as scope:
    sum_over_col = tf.reduce_sum(cm, axis=1)  # axes are switched for the total_cm within MeanIoU
    sum_over_row = tf.reduce_sum(cm, axis=0)  # axes are switched for the total_cm within MeanIoU
    tp = tf.linalg.diag_part(cm)
    fp = sum_over_row - tp
    fn = sum_over_col - tp
    iou = tf.math.divide_no_nan(tp, tp + fp + fn)
    recall = tf.math.divide_no_nan(tp, tp + fn)
    precision = tf.math.divide_no_nan(tp, tp + fp)
  return iou, recall, precision


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def normalize(x):
  return (x - x.min()) / (x.max() - x.min())
