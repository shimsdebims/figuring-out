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
    """Load model with comprehensive error handling"""
    model_path = "Model/crop_model.h5"
    
    # Get absolute path and verify existence
    abs_path = os.path.abspath(model_path)
    logger.info(f"Looking for model at: {abs_path}")
    
    if not os.path.exists(abs_path):
        # Try alternative paths if primary doesn't exist
        alternative_paths = [
            os.path.join(os.path.dirname(__file__), "Model/crop_model.h5"),
            os.path.join(os.getcwd(), "Model/crop_model.h5"),
            "crop_model.h5",
            "../Model/crop_model.h5"
        ]
        
        for path in alternative_paths:
            abs_path = os.path.abspath(path)
            logger.info(f"Trying alternative path: {abs_path}")
            if os.path.exists(abs_path):
                model_path = abs_path
                break
        else:
            raise FileNotFoundError(
                f"Could not find model file. Checked paths:\n"
                f"- {os.path.abspath('Model/crop_model.h5')}\n"
                f"- {os.path.join(os.path.dirname(__file__), 'Model/crop_model.h5')}\n"
                f"- {os.path.join(os.getcwd(), 'Model/crop_model.h5')}\n"
                f"Current directory contents: {os.listdir()}"
            )
    
    try:
        # Disable TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
        logger.info(f"Attempting to load model from: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Try loading with custom objects if needed
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=None
            )
            return model
        except Exception as e2:
            logger.error(f"Failed with custom objects: {str(e2)}")
            raise RuntimeError(f"Could not load model: {str(e2)}")

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
        # Load model (cached in session state)
        if 'model' not in st.session_state:
            st.session_state.model = load_model()
        model = st.session_state.model
        
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
        raise RuntimeError(f"Prediction error: {str(e)}")