#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"

docker run \
--gpus all \
--name='pcl_segmentation' \
--rm \
--tty \
--user "$(id -u):$(id -g)" \
--volume $DIR/../:/src \
pcl_segmentation \
python3 src/pcl_segmentation/inference.py \
--input_path="/src/sample_dataset/train/*.npy" \
--output_dir="/src/output/prediction" \
--path_to_model="/src/output/model" \
--model=squeezesegv2