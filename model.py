import os
import numpy as np
from PIL import Image
import logging
import tensorflow as tf
import cv2
import json
from pathlib import Path
from tensorflow import keras
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load class names
try:
    with open("disease_info.json", "r") as f:
        disease_info = json.load(f)
        CLASS_NAMES = list(disease_info.keys())
except Exception as e:
    logger.error(f"Error loading disease info: {str(e)}")
    CLASS_NAMES = ["Healthy"]

def load_model():
    model_path = 'Model/crop_model.h5'
    
    try:
        # First try standard load
        return tf.keras.models.load_model(model_path)
    except Exception as e1:
        try:
            # Try loading as weights only
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e2:
            try:
                # Try loading architecture + weights separately
                model = build_model_architecture()  # Define your model structure
                model.load_weights(model_path)
                return model
            except Exception as e3:
                raise RuntimeError(
                    f"All loading methods failed:\n"
                    f"1. Standard: {str(e1)}\n"
                    f"2. Weights Only: {str(e2)}\n"
                    f"3. Architecture + Weights: {str(e3)}"
                )

def is_plant_image(image):
    """Verify image contains plant material"""
    try:
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.mean(mask > 0)
        return green_percentage > 0.25
    except Exception as e:
        logger.warning(f"Plant verification failed: {str(e)}")
        return False

def predict_disease(image_input):
    """Make prediction with the loaded model"""
    try:
        # Open image
        img = Image.open(image_input).convert('RGB')
        
        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        model = load_model()  # Load fresh each time for reliability
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
        return disease, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "Error", 0.0