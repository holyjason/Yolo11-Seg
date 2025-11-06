# Yolo11 Segmentation

- Bill Liu Nov 2025

[TOC]

## Environment

- Tested on Jetson AGX Orin (64g version)
- Jetpack 6.2.1
- CUDA 12.6
- Using RealSense D415 and D455 Stereo Camera

## Usage

```
cd .
python3 main.py 
```

## File Structure


```
project/
├── main.py
├── camera/
│   ├── __init__.py
│   └── realsense_d455.py
├── segmentation/
│   ├── __init__.py
│   ├── yolov11_segmentation.py
│   └── segmentation_visualizer.py
├── utils/
│   ├── __init__.py
│   └── fps_counter.py
├── Weights/
│   ├── weight.md
│   ├── yolo11x-seg.engine
│   └── yolo11x-seg.pt
├── config/
│   └── setup.md
└── app/
    ├── __init__.py
    └── instance_segmentation_app.py
```

