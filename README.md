 # RPSGO

This project is a real-time, two-player Rock Paper Scissors game that uses computer vision and deep learning to detect hand gestures. It is specifically optimized to leverage an NVIDIA GPU (via CUDA) for low-latency inference, ensuring a smooth and responsive gaming experience.

The core logic is implemented as a state machine to manage the countdown, detection, and result display, preventing false starts and premature results.

## Features

**GPU Acceleration**: Utilizes PyTorch/CUDA for high-speed YOLOv5/v8 model inference on the GPU ('cuda:0').

**Real-Time Detection**: Classifies 'rock', 'paper', and 'scissors' gestures in real-time using a pre-trained custom model.

**Two-Player Game Screen**: Splits the webcam feed to track gestures from Player 1 (left side) and Player 2 (right side).

**Game State Management**: Implements a state machine for structured round-based gameplay.

**Performance Optimization**: Runs inference on a scaled-down image (640x480) for speed, then scales results back to the full display resolution.

**Live Scoreboard**: Tracks and displays scores for P1, P2, and Draws.

## Prerequisites

To run this application, you must have the following installed:

***Python 3.8+***

***Webcam***

## Required Libraries:

1.PyTorch (with CUDA support is highly recommended)

2.OpenCV

3.Ultralytics (YOLO framework)**

## Installation and Setup

**1. Clone the Repository**

git clone [https://github.com/L-Karthik-G/RPSGO.git](https://github.com/L-Karthik-G/RPSGO)

**2. Install Dependencies**

.Install required Python packages
  pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)  *Use the correct CUDA version for your system*
  pip install ultralytics opencv-python


**3. Obtain Model Weights**

**Crucial Step**: This application requires the trained YOLO model weights (best.pt).

**Model Path**: The code is configured to look for the weights at: runs/detect/rps_optimized/weights/best.pt

**Action**: Place your trained model file (or a placeholder if you intend to train it yourself) at the corresponding location within the repository structure.

## Running the Game

Ensure your webcam is connected and the dependencies are installed.

***python rps.py***


## Controls
1.**SPACE** --> *Start Round*

2.**R** --> *Reset Score*

3.**Q** --> *Quit*


