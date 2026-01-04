#!/bin/bash

sudo apt update
sudo apt install -y python3-picamera2
sudo apt install -y python3-opencv
sudo apt install -y libsndfile1
sudo apt install -y libportaudio2
sudo apt install -y pulseaudio

python3 -m pip config set global.break-system-packages true

pip install -U piper-tts sounddevice numpy
pip install -U pillow torch
pip install -U "huggingface-hub<1.0,>=0.34.0"
pip install -U transformers

mkdir -p /home/pi/models

export PATH=$PATH:~/.local/bin
huggingface-cli download dennisjooo/Birds-Classifier-EfficientNetB2 \
  --local-dir /home/pi/models/Birds-Classifier-EfficientNetB2 \
  --local-dir-use-symlinks False

wget https://huggingface.co/csukuangfj/vits-piper-en_US-lessac-medium/resolve/main/en_US-lessac-medium.onnx -O /home/pi/models/en_US-lessac-medium.onnx
wget https://huggingface.co/csukuangfj/vits-piper-en_US-lessac-medium/resolve/main/en_US-lessac-medium.onnx.json -O /home/pi/models/en_US-lessac-medium.onnx.json

sudo sh -c 'echo "camera_auto_detect=0" >> /boot/firmware/config.txt'
sudo sh -c 'echo "dtoverlay=imx708,cam0" >> /boot/firmware/config.txt'
sudo sh -c 'echo "dtoverlay=imx708,cam1" >> /boot/firmware/config.txt'

echo "amixer -D pulse sset Master 100%" >> /home/pi/.bashrc

sudo ./install-service.sh

echo "Setup complete! Rebooting your Raspberry Pi to apply camera settings."

sudo reboot