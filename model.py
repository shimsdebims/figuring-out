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
    """Robust model loading with multiple fallbacks"""
    try:
        model_path = Path(__file__).parent / "Model" / "crop_model.h5"
        
        # Basic validation
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        file_size = model_path.stat().st_size
        if file_size < 1024:  # Less than 1KB
            raise ValueError(f"Model file too small ({file_size} bytes), likely corrupted")
        
        # First try standard load
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            # Try with different options if standard load fails
            try:
                return tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=None
                )
            except Exception as e2:
                # Final attempt with experimental features
                try:
                    return tf.keras.models.load_model(
                        model_path,
                        compile=False,
                        custom_objects=None,
                        options=tf.saved_model.LoadOptions(
                            experimental_io_device='/job:localhost'
                        )
                    )
                except Exception as e3:
                    raise RuntimeError(
                        f"All loading attempts failed:\n"
                        f"1. Standard: {str(e)}\n"
                        f"2. With options: {str(e2)}\n"
                        f"3. Experimental: {str(e3)}"
                    )
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

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