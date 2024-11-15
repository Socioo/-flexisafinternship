import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_data(data_dir, image_size=(128, 128)):
    images = []
    for filename in os.listdir(data_dir):
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)  # Resize to uniform size
        img = img / 255.0  # Normalize pixel values
        images.append(img_to_array(img))  # Convert to array
    return np.array(images)