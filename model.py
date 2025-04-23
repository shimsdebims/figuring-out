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
from io import BytesIO

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

def is_plant_image(image):
    """Verify image contains plant material using color analysis"""
    try:
        # Convert various input types to numpy array
        if isinstance(image, bytes):
            img_array = np.array(Image.open(BytesIO(image)))
        elif hasattr(image, 'read'):
            img_array = np.array(Image.open(image))
        else:
            img_array = np.array(Image.open(BytesIO(image.getvalue())))
            
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

@st.cache_resource
def load_model():
    """Load model with Streamlit caching"""
    MODEL_URL = "https://github.com/yourusername/yourrepo/releases/download/v1.0/plant_disease_model.zip"
    MODEL_PATH = "Model/plant_disease_model.h5"
    
    try:
        # Check if model exists locally
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)
            
        # Download and extract if not
        os.makedirs("Model", exist_ok=True)
        with st.spinner("Downloading model..."):
            # Download logic here (use urllib or requests)
            # Extract the ZIP file
            # Save to MODEL_PATH
            
            return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return MockModel()

def predict_disease(image_input):
    """Robust prediction function"""
    try:
        # Convert various input types to PIL Image
        if isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        elif hasattr(image_input, 'read'):
            img = Image.open(image_input)
        else:
            img = Image.open(BytesIO(image_input.getvalue()))
            
        img = img.convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        model = st.session_state.model
        predictions = model.predict(img_array, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error", 0.0