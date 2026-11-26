
# VisionTrack AI – AI-Powered Object Detection & Tracking System

VisionTrack AI is a computer vision application that detects objects in images and tracks objects in videos using YOLO and OpenCV. It provides a simple web interface built with Flask for uploading images and videos and viewing the detection results.

## Features

- Object detection in images
- Multiple image testing
- Video object tracking
- Bounding boxes and object labels
- Tracking IDs for detected objects in videos
- Detection statistics
- Web interface built with Flask
- Pretrained YOLO model
- CPU-based inference

## Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- Flask
- NumPy
- HTML
- CSS

## Project Structure

VisionTrack-AI/
├── app.py
├── yolo26n.pt
├── requirements.txt
├── test_model.py
├── test_multiple_images.py
├── README.md
├── .gitignore
├── templates/
│   └── index.html
├── static/
│   └── results/
├── uploads/
├── test_images/
└── outputs/

## Installation

1. Clone the repository:

   git clone https://github.com/YOUR-USERNAME/VisionTrack-AI.git

2. Navigate to the project directory:

   cd VisionTrack-AI

3. Create a virtual environment:

   python -m venv .venv

4. Activate the virtual environment:

   .venv\Scripts\activate

5. Install dependencies:

   pip install -r requirements.txt

## Run the Application

Run the Flask application:

python app.py

Open your browser and visit:

http://127.0.0.1:5000

Upload an image or video to detect and track objects.

## Model

The project uses the pretrained YOLO26n model for object detection and tracking.

## Applications

- Traffic monitoring
- Surveillance
- Object identification
- Video analysis
- Computer vision applications

## Author

Kundana Sree

## Project

Developed as part of the CodeAlpha Internship.
