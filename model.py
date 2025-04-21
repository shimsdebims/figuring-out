"""
Plant Disease Prediction Model
Handles image classification using TensorFlow/Keras
"""

import os
import numpy as np
from PIL import Image
import logging
import json
import cv2
import streamlit as st
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load class names from JSON
try:
    with open("disease_info.json") as f:
        disease_info = json.load(f)
        CLASS_NAMES = list(disease_info.keys())
except Exception as e:
    logger.error(f"Error loading disease info: {str(e)}")
    CLASS_NAMES = ["Healthy"]

class MockModel:
    """Fallback mock model if real model fails to load"""
    def predict(self, x):
        return np.random.uniform(0, 1, size=(1, len(CLASS_NAMES)))

def load_model():
    """Load the trained TensorFlow model with error handling"""
    try:
        model_paths = [
            "Model/crop_model.h5",
            "Model/plant_disease_model.h5"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return tf.keras.models.load_model(path)
        
        logger.warning("No model file found, using mock model")
        return MockModel()
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return MockModel()

def is_plant_image(image):
    """Verify image contains plant material using color analysis"""
    try:
        img_array = np.array(image)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return False
            
        # Convert to HSV and detect green pixels
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
    """Make disease prediction on input image"""
    try:
        # Preprocess image
        img = Image.open(image_input).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get model from session state
        model = st.session_state.model
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
        # Handle mock model case
        if isinstance(model, MockModel):
            disease = "[MOCK] " + disease
            confidence = min(confidence, 0.95)
            
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "Error", 0.0