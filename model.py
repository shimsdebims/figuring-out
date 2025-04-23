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
    try:
        model_url = "https://drive.google.com/uc?export=download&id=1vljcvW2gO_uU88igu41dG8MlKN1quoVP"
        model_path = "Model/plant_disease_model.h5"
        
        # Download only if file doesn't exist
        if not os.path.exists(model_path):
            os.makedirs("Model", exist_ok=True)
            
            # Download model
            response = requests.get(model_url)
            response.raise_for_status()  # Check for errors
            
            # Save to disk
            with open(model_path, "wb") as f:
                f.write(response.content)
            
            print("✅ Model downloaded from Google Drive!")
        
        # Load .h5 model
        return tf.keras.models.load_model(model_path)
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
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
    try:
        # Preprocess image
        img = Image.open(image_input).convert('RGB')
        img = img.resize((224, 224))  # Match your model's expected input
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get model from session state
        model = st.session_state.model
        
        # Predict (for .h5 model)
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "Error", 0.0