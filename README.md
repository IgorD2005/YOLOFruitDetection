YOLOv8 Fruit Detection

A real-time object detection system built on YOLOv8, fine-tuned to detect four fruit categories in images and video streams. Developed as part of a computer vision course.

Model

YOLOv8n (nano) was selected as the base model and fine-tuned on a custom fruit dataset sourced from Roboflow. Training ran for up to 20 epochs with a confidence threshold of 0.5 applied during inference. The fine-tuned weights are saved to runs/detect/train/weights/best.pt upon completion.
Classes: apple, banana, pineapple, strawberry

Features

Detection is supported across three input modes: static images, video files, and live camera streams. A side-by-side comparison mode allows the base and fine-tuned models to be evaluated on the same input simultaneously. Bounding boxes, class labels, and confidence scores are rendered directly onto each frame, with a per-class colour scheme generated deterministically from the class index.

Dataset

Annotation was performed in YOLO format, with each label file containing normalised bounding box coordinates alongside a class identifier. The dataset configuration is defined in data.yaml and follows the standard Ultralytics directory structure, with separate image and label folders for training and validation subsets.

Project Structure

lab6.py - full pipeline: detection, training, comparison, and video processing

data.yaml - dataset configuration for Ultralytics

runs/detect/train/ - training outputs including metrics and visualisations

.gitignore - excludes model weights and raw image data

Requirements:
ultralytics,
opencv-python,
numpy.

Usage

Detection on image: python lab6.py --model yolov8n.pt --image path/to/image.jpg
Detection from camera: python lab6.py --model yolov8n.pt --camera
Fine-tuning: python lab6.py --model yolov8n.pt --train --train-data dataset/data.yaml
Compare models: python lab6.py --model yolov8n.pt --trained-model best.pt --compare --image path/to/image.jpg
