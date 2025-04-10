import os
import numpy as np
from PIL import Image
import logging
import tensorflow as tf
import cv2
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load class names
disease_info_path = "disease_info.json"
if not os.path.exists(disease_info_path):
    logger.error(f"Disease info file not found at {disease_info_path}")
    # Create a basic default
    disease_info = {"Healthy": {"symptoms": "None", "treatment": "None", "fun_fact": "Healthy plants are happy plants!"}}
    CLASS_NAMES = ["Healthy"]
else:
    try:
        with open(disease_info_path, "r") as f:
            disease_info = json.load(f)
            CLASS_NAMES = list(disease_info.keys())
            logger.info(f"Loaded {len(CLASS_NAMES)} disease classes")
    except Exception as e:
        logger.error(f"Error loading disease info: {str(e)}")
        # Create a basic default
        disease_info = {"Healthy": {"symptoms": "None", "treatment": "None", "fun_fact": "Healthy plants are happy plants!"}}
        CLASS_NAMES = ["Healthy"]

def load_model():
    """Load model from the correct location"""
    # Exact path where your model is located
    model_path = os.path.join("Models", "crop_model.h5")
    
    logger.info(f"Attempting to load model from: {model_path}")
    
    if os.path.exists(model_path):
        try:
            # Disable TensorFlow logging
            tf.get_logger().setLevel('ERROR')
            tf.autograph.set_verbosity(0)
            
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    else:
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please check that this path is correct.")
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Disable TensorFlow logging
                tf.get_logger().setLevel('ERROR')
                tf.autograph.set_verbosity(0)
                
                model = tf.keras.models.load_model(path, compile=False)
                logger.info(f"Successfully loaded model from {path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {str(e)}")
    
    # If we reach here, no model was successfully loaded
    logger.error("No valid model found in any of the searched locations")
    # Return a dummy model or raise an exception
    raise FileNotFoundError("No valid model found. Please check the model file path.")

def is_plant_image(image):
    """Verify image contains plant material"""
    try:
        img_array = np.array(image)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define green color range
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Threshold the HSV image
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.mean(mask > 0)
        
        return green_percentage > 0.25  # At least 25% green pixels
    except Exception as e:
        logger.warning(f"Plant verification failed: {str(e)}")
        return False

def predict_disease(image_input):
    """Make prediction with the loaded model"""
    try:
        model = load_model()
        
        # Open image
        if isinstance(image_input, (str, os.PathLike)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = Image.open(image_input).convert('RGB')
        
        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        # Get class name
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
        return disease, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise