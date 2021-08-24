#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=2

docker run \
--gpus all \
--name='pcl_segmentation' \
--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
--rm \
--tty \
--user "$(id -u):$(id -g)" \
--volume $DIR/../:/src \
pcl_segmentation \
python3 src/pcl_segmentation/train.py \
--data_path="src/sample_dataset" \
--train_dir="src/output" \
--epochs=5 \
--model=squeezesegv2