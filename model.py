import os
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import h5py
import json
import cv2
import streamlit as st
import tensorflow as tf 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load class names
try:
    with open("disease_info.json", "r") as f:
        disease_info = json.load(f)
        CLASS_NAMES = list(disease_info.keys())
    logger.info(f"Loaded {len(CLASS_NAMES)} disease classes")
except Exception as e:
    logger.error(f"Error loading disease info: {str(e)}")
    CLASS_NAMES = ["Healthy"]

def load_model():
    model_path = Path("Model/plant_disease_model.h5")
    
    # Verify file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path.absolute()}")
    
    # Verify file is not empty
    if model_path.stat().st_size == 0:
        raise ValueError("Model file is empty")
    
    logger.info(f"Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Try loading with different options
        try:
            model = tf.keras.models.load_model(str(model_path))
            logger.info("Model loaded successfully with default options")
            return model
        except Exception as e:
            logger.warning(f"Standard load failed, trying with custom objects: {str(e)}")
            model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects=None
            )
            logger.info("Model loaded successfully with custom options")
            return model
            
    except Exception as e:
        logger.error(f"Critical error loading model: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

def is_plant_image(image):
    """Verify image contains plant material"""
    try:
        img_array = np.array(image)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return False
            
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.mean(mask > 0)
        return green_percentage > 0.25
    except Exception as e:
        logger.warning(f"Plant verification failed: {str(e)}")
        return False

# def predict_disease(image_input):
#     """Make prediction with the loaded model"""
#     try:
#         # Open image
#         img = Image.open(image_input).convert('RGB')
        
#         # Preprocess
#         img = img.resize((224, 224))
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
        
#         # Get model
#         import tensorflow as tf
        
#         # Load model if not in session state
#         if not hasattr(st.session_state, 'model') or st.session_state.model is None:
#             st.session_state.model = load_model()
#         model = st.session_state.model
        
#         # Predict
#         predictions = model.predict(img_array, verbose=0)[0]
#         predicted_idx = np.argmax(predictions)
#         confidence = float(predictions[predicted_idx])
#         disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
#         return disease, confidence
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         return "Error", 0.0
    

def predict_disease(image_input):
    """Make prediction with the loaded model (temporary mock for presentation)"""
    try:
        # Open image just to check it's valid
        img = Image.open(image_input).convert('RGB')
        
        # Mock prediction - replace with your model's classes as needed
        import random
        diseases = ["Tomato - Healthy", "Tomato - Leaf Mold", "Potato - Late Blight"]
        disease = random.choice(diseases)
        confidence = random.uniform(0.75, 0.98)
        
        return disease, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "Error", 0.0