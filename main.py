import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from tensorflow.keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model.savedmodel", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame from camera. Check if camera is connected.")
        break

    # Resize the raw image into (224-height,224-width) pixels
    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image_input = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_input = (image_input / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image_input)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end=" ")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Draw text on the image
    text = f"{class_name[2:]}: {str(np.round(confidence_score * 100))[:-2]}%"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
