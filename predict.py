import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import os

# Load Trained Model that you saves after training

model = keras.models.load_model("cat_dog_model.h5")


# Load and Preprocess Image

def preprocess_image(img_path):
    img = keras.utils.load_img(img_path, target_size=(256, 256))  # resize
    img_array = keras.utils.img_to_array(img) / 255.0            # normalize
    img_array = np.expand_dims(img_array, axis=0)                # batch dimension
    return img_array


# Predict

def predict(img_path):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]
    
    if prediction < 0.5:
        print(f"Prediction: Cat ({(1 - prediction) * 100:.2f}% confidence)")
    else:
        print(f"Prediction: Dog ({prediction * 100:.2f}% confidence)")


# CLI Support

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict(sys.argv[1])
