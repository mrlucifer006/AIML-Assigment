import os

# 1. Enable Legacy Keras Support for TF 2.x
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import random
import time

app = FastAPI()

# 2. Load Model and Labels
try:
    print("Loading model...")
    model = load_model("model.savedmodel", compile=False)
    print("Model loaded successfully.")
    
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Labels loaded: {class_names}")
except Exception as e:
    print(f"Error loading model/labels: {e}")
    class_names = []

templates = Jinja2Templates(directory="templates")

# Choice Map
# Assuming labels.txt is: "0 Rock", "1 Scissor", "2 Paper" (Check this execution)
# Based on labels.txt content: "0 rock", "1 Sicssor", "2 paper"
# Standardize names for game logic: "rock", "paper", "scissors"
NORMALIZED_LABELS = {
    "rock": "rock",
    "paper": "paper",
    "sicssor": "scissors", # Fix typo from labels.txt
    "scissor": "scissors",
    "scissors": "scissors"
}

def get_computer_choice():
    return random.choice(["rock", "paper", "scissors"])

def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "It's a Tie!"
    
    winning_combos = {
        "rock": "scissors",
        "paper": "rock",
        "scissors": "paper"
    }

    if winning_combos.get(user_choice) == computer_choice:
        return "You Win!"
    else:
        return "Computer Wins!"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/play")
async def play(file: UploadFile = File(...)):
    # 3. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 4. Preprocess Image
    # Resize to (224, 224)
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Reshape and Normalize
    image_input = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_input = (image_input / 127.5) - 1

    # 5. Predict
    prediction = model.predict(image_input)
    index = np.argmax(prediction)
    raw_label = class_names[index] # e.g., "0 rock"
    confidence = float(prediction[0][index])

    # Extract class name (remove index "0 ", "1 ", etc.)
    user_move_raw = raw_label[2:].lower().strip() # "rock", "sicssor", "paper"
    user_move = NORMALIZED_LABELS.get(user_move_raw, user_move_raw)

    # 6. Game Logic
    computer_move = get_computer_choice()
    result = determine_winner(user_move, computer_move)

    return {
        "user_move": user_move,
        "computer_move": computer_move,
        "result": result,
        "confidence": f"{confidence * 100:.1f}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
