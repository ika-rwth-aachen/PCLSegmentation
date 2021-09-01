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
import argparse

from data_loader import DataLoader
from utils.util import confusion_matrix_to_iou_recall_precision
from utils.args_loader import load_model_config


def evaluation(arg):
  config, _ = load_model_config(arg.model)

  config.DATA_AUGMENTATION = False
  config.BATCH_SIZE = 1
  dataset = DataLoader(arg.image_set, arg.data_path, config).write_tfrecord_dataset().read_tfrecord_dataset()

  model = tf.keras.models.load_model(arg.path_to_model)
  miou_tracker = tf.metrics.MeanIoU(num_classes=config.NUM_CLASS, name="MeanIoU")

  print("Performing Evaluation")

  for sample in dataset:
    (lidar, mask), label, weight = sample
    probabilities, predictions = model([lidar, mask])
    miou_tracker.update_state(label, predictions)

  iou, recall, precision = confusion_matrix_to_iou_recall_precision(miou_tracker.total_cm)

  for i, cls in enumerate(config.CLASSES):
    print(cls.upper())
    print("IoU:       " + str(iou[i].numpy()))
    print("Recall:    " + str(recall[i].numpy()))
    print("Precision: " + str(precision[i].numpy()))
    print("")
  print("MIoU: {} ".format(miou_tracker.result().numpy()))


if __name__ == '__main__':
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

  parser = argparse.ArgumentParser(description='Parse Flags for the evaluation script!')
  parser.add_argument('-d', '--data_path', type=str,
                      help='Absolute path to the dataset')
  parser.add_argument('-i', '--image_set', type=str, default="val",
                      help='Default: `val`. But can also be train, val or test')
  parser.add_argument('-t', '--eval_dir', type=str,
                      help="Directory where to write the Tensorboard logs and checkpoints")
  parser.add_argument('-p', '--path_to_model', type=str,
                      help='Path to the model')
  parser.add_argument('-m', '--model', type=str,
                      help='Model name either `squeezesegv2`, `darknet53`, `darknet21`')
  args = parser.parse_args()

  evaluation(args)
