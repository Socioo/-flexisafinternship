import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_data(data_dir, image_size):
    images = []
    for img_file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip if the image can't be read
        img = cv2.resize(img, (image_size[1], image_size[0]))  # Resize to width x height
        img = img / 255.0  # Normalise the image
        images.append(img)
    return np.array(images)