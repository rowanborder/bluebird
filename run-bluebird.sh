#!/bin/bash

amixer -D pulse sset Master 100%

/usr/bin/python3 /home/pi/bluebird/bluebird_cam.py \
    --model-dir /home/pi/models/Birds-Classifier-EfficientNetB2 \
    --piper-model /home/pi/models/en_US-lessac-medium.onnx \
    --host 0.0.0.0 --port 8000