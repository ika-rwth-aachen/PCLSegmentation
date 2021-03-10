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

from utils.util import plot_confusion_matrix, confusion_matrix_to_iou_recall_precision, plot_to_image


class TensorBoard(tf.keras.callbacks.TensorBoard):
  """Callback for storing the intermediate results of the point cloud segmentation networks"""

  def __init__(self, log_dir, dataset, **kwargs):
    super().__init__(log_dir, **kwargs)
    self.dataset = dataset
    self.num_images = 1
    self.custom_tb_writer = tf.summary.create_file_writer(self.log_dir + '/validation')

  def on_train_batch_end(self, batch, logs=None):
    lr = getattr(self.model.optimizer, 'lr', None)
    steps = self.model.optimizer.iterations
    with self.custom_tb_writer.as_default():
      tf.summary.scalar('step_learning_rate', lr(steps), steps)
    super().on_train_batch_end(batch, logs)

  def on_epoch_end(self, epoch, logs=None):
    batch_size = self.model.BATCH_SIZE
    class_color_map = self.model.CLS_COLOR_MAP

    # get first batch of dataset
    (lidar_input, lidar_mask), label, _ = self.dataset[0]

    probabilities, predictions = self.model([lidar_input, lidar_mask])

    label = label[:self.num_images, :, :]
    predictions = predictions[:self.num_images, :, :].numpy()

    # label and prediction visualizations
    label_image = class_color_map[label.reshape(-1)].reshape([self.num_images, label.shape[1], label.shape[2], 3])
    pred_image = class_color_map[predictions.reshape(-1)].reshape([self.num_images, label.shape[1], label.shape[2], 3])

    # confusion matrix visualization
    figure = plot_confusion_matrix(self.model.miou_tracker.total_cm.numpy(),
                                   class_names=self.model.mc.CLASSES)
    cm_image = plot_to_image(figure)

    with self.custom_tb_writer.as_default():
      tf.summary.image('Images/Depth Image',
                       lidar_input[:self.num_images, :, :, [4]],
                       max_outputs=batch_size,
                       step=epoch)
      tf.summary.image('Images/Label Image',
                       label_image,
                       max_outputs=batch_size,
                       step=epoch)
      tf.summary.image('Images/Prediction Image',
                       pred_image,
                       max_outputs=batch_size,
                       step=epoch)
      tf.summary.image("Confusion Matrix",
                       cm_image,
                       step=epoch)

    # Save IoU, Precision, Recall
    iou, recall, precision = confusion_matrix_to_iou_recall_precision(self.model.miou_tracker.total_cm)
    with self.custom_tb_writer.as_default():
      for i, cls in enumerate(self.model.mc.CLASSES):
        tf.summary.scalar('IoU/'+cls, iou[i], step=epoch)
        tf.summary.scalar('Recall/'+cls, recall[i], step=epoch)
        tf.summary.scalar('Precision/'+cls, precision[i], step=epoch)

    super().on_epoch_end(epoch, logs)

  def on_test_end(self, logs=None):
    super().on_test_end(logs)

  def on_train_end(self, logs=None):
    super().on_train_end(logs)
    self.custom_tb_writer.close()