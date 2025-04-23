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
    MODEL_URL = "https://drive.google.com/uc?id=1vljcvW2gO_uU88igu41dG8MlKN1quoVP&export=download"
    MODEL_PATH = "Model/plant_disease_model.h5"
    
    try:
        # 1. Check if model exists locally
        if os.path.exists(MODEL_PATH):
            logger.info("Loading model from local cache")
            return tf.keras.models.load_model(MODEL_PATH)
            
        # 2. Download from Google Drive if not found locally
        os.makedirs("Model", exist_ok=True)
        with st.spinner("Downloading model (this may take a few minutes)..."):
            import requests
            import gdown
            
            # Method 1: Using gdown (best for Google Drive)
            output = "Model/model_temp.h5"
            gdown.download(MODEL_URL, output, quiet=False)
            
            # Verify download completed
            if os.path.exists(output):
                os.rename(output, MODEL_PATH)
                return tf.keras.models.load_model(MODEL_PATH)
            
            # Fallback method using requests (if gdown fails)
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return tf.keras.models.load_model(MODEL_PATH)
                
        raise Exception("Download failed - no model file received")
        
    except Exception as e:
        logger.error(f"Model load failed: {str(e)}")
        st.error("⚠️ Failed to load disease detection model. Using limited demo mode.")
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