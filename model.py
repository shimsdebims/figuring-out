import os
import numpy as np
from PIL import Image
import logging
import tensorflow as tf
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Single model path
MODEL_PATH = os.path.join("Model", "crop_model.h5")  # Update with your model filename

# Load class names
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)
    CLASS_NAMES = list(disease_info.keys())

def load_model():
    """Load only the specified model"""
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    try:
        # Disable TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

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