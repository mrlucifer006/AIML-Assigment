# AI Rock-Paper-Scissors

This project features an AI-powered Rock-Paper-Scissors game that uses a trained machine learning model to recognize the player's hand gestures (Rock, Paper, or Scissors) from images or a live webcam feed.

## Features

- **Web Application (`app.py`)**: A FastAPI based web server that provides an interface to upload an image of a hand gesture. It predicts the gesture and plays a round of Rock-Paper-Scissors against the computer.
- **Real-Time Webcam (`main.py`)**: A desktop script using OpenCV to capture live video from your webcam and predict your hand gesture in real time.

## Prerequisites

Make sure you have Python installed. You will need the following libraries:

- `fastapi`
- `uvicorn`
- `tensorflow` (or `tensorflow-cpu`)
- `opencv-python`
- `numpy`
- `jinja2`
- `python-multipart`

You can install all dependencies using pip:
```bash
pip install fastapi uvicorn tensorflow opencv-python numpy jinja2 python-multipart
```

## How to Run

### 1. Web Application (`app.py`)

To run the web application, execute the following command in the project directory:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then, open your web browser and go to `http://localhost:8000`.

### 2. Live Webcam Version (`main.py`)

To test the model with your live webcam feed, run:

```bash
python main.py
```
A window will open showing your webcam feed with real-time predictions. Press `ESC` to close the window.

## Project Structure

- `app.py` - FastAPI backend logic for the web application.
- `main.py` - OpenCV real-time prediction script.
- `model.savedmodel/` - The pre-trained TensorFlow/Keras model directory.
- `labels.txt` - Text file containing the mapping of output classes.
- `templates/` - HTML templates for the FastAPI web app.

## Note

Ensure that your webcam is connected and the execution environment has enough memory to load the TensorFlow model.
