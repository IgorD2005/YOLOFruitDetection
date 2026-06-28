# YOLOv8 Fruit Detection

A real-time object detection system built on **YOLOv8**, fine-tuned to detect four fruit categories in images and video streams. Developed as part of a computer vision course.

---

## Model

The project uses **YOLOv8n (Nano)** as the base model, which was fine-tuned on a custom fruit dataset sourced from **Roboflow**.

### Training Configuration

- **Base Model:** YOLOv8n
- **Training Epochs:** Up to 20
- **Confidence Threshold:** `0.5` (during inference)
- **Output Weights:** `runs/detect/train/weights/best.pt`

### Detected Classes

- 🍎 Apple
- 🍌 Banana
- 🍍 Pineapple
- 🍓 Strawberry

---

## Features

The application supports object detection in multiple input modes:

- 🖼️ Static images
- 🎥 Video files
- 📷 Live camera stream

Additional functionality includes:

- Side-by-side comparison between the **base** and **fine-tuned** models
- Real-time rendering of:
  - Bounding boxes
  - Class labels
  - Confidence scores
- Deterministic per-class color generation based on class index

---

## Dataset

The dataset follows the standard **Ultralytics YOLO** directory structure.

Annotations are stored in **YOLO format**, where each label file contains:

- Class identifier
- Normalized bounding box coordinates

The dataset configuration is defined in:

```text
data.yaml
```

Dataset structure:

- `train/images`
- `train/labels`
- `valid/images`
- `valid/labels`

---

## Project Structure

```text
.
├── lab6.py                     # Detection, training, comparison, and video processing
├── data.yaml                   # Dataset configuration
├── runs/
│   └── detect/
│       └── train/              # Training outputs, metrics, and visualizations
└── .gitignore                  # Excludes model weights and raw datasets
```

---

## Requirements

- Ultralytics
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

---

## Usage

### Detect objects in an image

```bash
python lab6.py --model yolov8n.pt --image path/to/image.jpg
```

### Detect objects from a live camera

```bash
python lab6.py --model yolov8n.pt --camera
```

### Fine-tune the model

```bash
python lab6.py --model yolov8n.pt --train --train-data dataset/data.yaml
```

### Compare the base and fine-tuned models

```bash
python lab6.py --model yolov8n.pt --trained-model best.pt --compare --image path/to/image.jpg
```
