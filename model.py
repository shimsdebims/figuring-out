import os
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import h5py
import json
import cv2

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
    
    try:
        # Try TensorFlow loading first
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        logger.info("Model loaded with TensorFlow successfully")
        return model
    except Exception as e1:
        logger.warning(f"TensorFlow loading failed: {e1}")
        try:
            # Fallback to standalone Keras
            from keras.models import load_model as keras_load
            model = keras_load(str(model_path))
            logger.info("Model loaded with Keras successfully")
            return model
        except Exception as e2:
            logger.error(f"Keras loading failed: {e2}")
            raise RuntimeError(
                f"All loading methods failed:\n"
                f"TensorFlow: {e1}\n"
                f"Keras: {e2}"
            )

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

def predict_disease(image_input):
    """Make prediction with the loaded model"""
    try:
        # Open image
        img = Image.open(image_input).convert('RGB')
        
        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get model
        if 'model' not in st.session_state:
            st.session_state.model = load_model()
        model = st.session_state.model
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
        return disease, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "Error", 0.0