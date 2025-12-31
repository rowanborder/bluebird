#!/bin/zsh

sudo apt update
sudo apt install -y python3-picamera2
sudo apt install -y python3-opencv
sudo apt install -y libsndfile1


pip install -U piper-tts sounddevice numpy
pip install -U transformers pillow torch
pip install -U "huggingface-hub<1.0,>=0.34.0"

export PATH=$PATH:~/.local/bin
huggingface-cli download dennisjooo/Birds-Classifier-EfficientNetB2 \
  --local-dir /home/pi/models/Birds-Classifier-EfficientNetB2 \
  --local-dir-use-symlinks False

