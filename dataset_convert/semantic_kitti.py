#!/usr/bin/env python3
import os

import argparse
import numpy as np
import tqdm
import yaml

import matplotlib.pyplot as plt

from laserscan_semantic_kitti import SemLaserScan

CONFIG_FILE = "semantic-kitti.yaml"


# Layer extraction for VLP32-C
layers = np.arange(16, 48)

"""
label_map_to_ika = {
  # road
  60: 0,
  40: 0,
  44: 0,

  # sidewalk
  48: 1,

  # building
  50: 2,
  51: 2,
  52: 2,

  # pole
  80: 3,
  81: 3,

  # vegetation
  70: 4,
  71: 4,
  72: 4,

  # person
  30: 5,
  254: 5,
  32: 5,
  31: 5,

  # two-wheeler
  11: 6,
  15: 6,
  253: 6,
  255: 6,

  # car
  252: 7,
  20: 7,
  259: 7,
  10: 7,

  # truck
  18: 8,
  258: 8,

  # bus
  13: 9,
  257: 9,

  # none
  0: 10,
  1: 10,
  256: 10,
  16: 10,
  49: 10,
  99: 10
}
"""


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./semantic_kitti.py")
  parser.add_argument(
    '--dataset', '-d',
    type=str,
    required=True,
    help='path to the kitti dataset where the `sequences` directory is. No Default',
  )
  parser.add_argument(
    '--output_dir', '-p',
    type=str,
    required=True,
    help='output path to the PCL segmentation repository. No Default',
  )
  parser.add_argument(
    '-v',
    action='store_true',
    help='use only 32 layers of KITTI lidar data in order to match a VLP32 laser scanner ',
  )

  FLAGS, unparsed = parser.parse_known_args()

  # open config file
  print("Opening config file %s" % CONFIG_FILE)
  config = yaml.safe_load(open(CONFIG_FILE, 'r'))

  for split in ["train", "val", "test"]:
    sequences = config["split"][split]

    scans = []
    labels = []

    for sequence in sequences:
      # path which contains the pointclouds for each scan
      scan_paths = os.path.join(FLAGS.dataset, "sequences", str(sequence).zfill(2), "velodyne")

      if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
      else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

      scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
      scan_names.sort()
      scans = scans + scan_names

    for sequence in sequences:
      # path which contains the labels for each scan
      label_paths = os.path.join(FLAGS.dataset, "sequences", str(sequence).zfill(2), "labels")

      if os.path.isdir(label_paths):
        print("Labels folder exists! Using labels from %s" % label_paths)
      else:
        print("Labels folder doesn't exist! Exiting...")
        quit()

      label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn]
      label_names.sort()
      labels = labels + label_names

    print("number of scans : ", len(scans))
    print("number of labels : ", len(labels))

    # apply learning map
    learning_map = config["learning_map"]
    vfunc = np.vectorize(learning_map.get)

    color_dict = config["color_map"]  # maps numeric labels in .label file to a bgr color
    nclasses = len(color_dict)  # number of classes

    laser_scan = SemLaserScan(nclasses, color_dict, project=True)  # create a scan with all 64 layers of KITTI lidar data

    for index, (scan, label) in tqdm.tqdm(enumerate(zip(scans, labels)), total=len(scans)):
      laser_scan.open_scan(scan)
      laser_scan.open_label(label)

      if index == 0:
        plt.figure(figsize=(30, 3))
        plt.imshow(laser_scan.proj_sem_color)
        plt.tight_layout()
        plt.show()

      mask = laser_scan.proj_range > 0  # check if the projected depth is positive
      laser_scan.proj_range[~mask] = 0.0
      laser_scan.proj_xyz[~mask] = 0.0
      laser_scan.proj_remission[~mask] = 0.0
      laser_scan.proj_sem_label = vfunc(laser_scan.proj_sem_label)  # map class labels to values between 0 and 10

      # create the final data sample with shape (64,1024,6)
      final_data = np.concatenate([laser_scan.proj_xyz,
                                   laser_scan.proj_remission.reshape((64, 1024, 1)),
                                   laser_scan.proj_range.reshape((64, 1024, 1)),
                                   laser_scan.proj_sem_label.reshape((64, 1024, 1))],
                                  axis=2)

      if FLAGS.v:
        vlp_32_data = final_data[layers, :, :]
        np.save(os.path.join(FLAGS.output_dir, "converted_dataset", split, str(index)), vlp_32_data)
      else:
        np.save(os.path.join(FLAGS.output_dir, "converted_dataset", split, str(index)), final_data)
