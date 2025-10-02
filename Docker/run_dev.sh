#!/bin/bash

PROJECT_NAME="multi_view_stereo"
IMAGE_NAME="${PROJECT_NAME}:12.4.0-devel-ubuntu20.04"
DATA_PATH="/media/${USER}/zhipeng_usb/datasets"
DATA_PATH2="/media/${USER}/zhipeng_8t1/datasets"

# Pick up config image key if specified
if [[ ! -z "${CONFIG_DATA_PATH}" ]]; then
    DATA_PATH=$CONFIG_DATA_PATH
fi



docker build -t $IMAGE_NAME -f "${HOME}/vscode_projects/multi_view_stereo/Docker/Dockerfile.txt" .


xhost +local:root

docker run \
    --rm \
    -e DISPLAY=$DISPLAY \
    --shm-size=8g \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${HOME}/vscode_projects/multi_view_stereo/:/root/multi_view_stereo \
    -v ${DATA_PATH}:/root/datasets \
    -v ${DATA_PATH2}:/root/datasets2 \
    --privileged \
    --cap-add sys_ptrace \
    --runtime=nvidia \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

# docker run --rm -it --name $PROJECT_NAME $IMAGE_NAME /bin/bash

# xhost -local:root