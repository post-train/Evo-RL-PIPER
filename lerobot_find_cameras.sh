#!/bin/bash

# 中文简介：扫描并抓拍当前可用摄像头，帮助确认 OpenCV 相机索引、画面方向和接线是否正确。

python src/lerobot/scripts/lerobot_find_cameras.py opencv --output-dir outputs/captured_images --record-time-s 1
#python src/lerobot/scripts/lerobot_find_cameras.py realsense --output-dir outputs/captured_images
