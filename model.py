import os
import json
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import TensorFlow
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("TensorFlow successfully imported")
except ImportError as e:
    logger.warning(f"TensorFlow not available: {str(e)}")

# Load class names from disease info
try:
    with open("disease_info.json", "r") as f:
        disease_info = json.load(f)
        CLASS_NAMES = list(disease_info.keys())
    logger.info(f"Loaded {len(CLASS_NAMES)} disease classes")
except Exception as e:
    logger.error(f"Error loading disease_info.json: {str(e)}")
    CLASS_NAMES = ["Unknown"]

def load_model():
    """Load the crop disease detection model"""
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available - using dummy model")
        return "dummy_model"
    
    # Define possible model paths
    model_paths = [
        os.path.join("Model", "best_model.h5"),
        os.path.join("Model", "plant_disease_model.h5"),
        os.path.join("Model", "model.tflite"),
        os.path.join("model", "best_model.h5"),  # Try lowercase directory
        os.path.join("model", "plant_disease_model.h5"),
        os.path.join("model", "model.tflite")
    ]
    
    # Try each possible path
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"Found model at {model_path}")
            
            try:
                if model_path.endswith('.h5'):
                    # Disable TensorFlow logging
                    tf.get_logger().setLevel('ERROR')
                    tf.autograph.set_verbosity(0)
                    
                    # Load Keras model
                    model = tf.keras.models.load_model(model_path, compile=False)
                    logger.info(f"Successfully loaded Keras model from {model_path}")
                    
                    # Print model summary for verification
                    logger.info("Model summary:")
                    model.summary(print_fn=logger.info)
                    
                    return model
                
                elif model_path.endswith('.tflite'):
                    # Load TFLite model
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    logger.info(f"Successfully loaded TFLite model from {model_path}")
                    return interpreter
            
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                continue
    
    logger.error("No valid model found in any of the searched locations")
    return "dummy_model"

# Add to model.py
def is_plant_image(image):
    """Basic verification if image contains plant leaves"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check dominant color is green (plant-like)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
        green_percentage = np.mean(green_mask > 0)
        
        # Check texture (plants have more texture than faces)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return green_percentage > 0.3 or laplacian_var > 100  # Adjust thresholds
    except:
        return False

# Update predict_disease function
def predict_disease(image_input, model=None):
    # First verify it's a plant image
    if isinstance(image_input, (str, os.PathLike)):
        img = Image.open(image_input).convert('RGB')
    else:
        img = Image.open(image_input).convert('RGB')
    
    if not is_plant_image(img):
        return "Not a plant image", 0.0
    

def predict_disease(image_path, model=None):
    """Predict plant disease from an image"""
    try:
        # Load model if not provided
        if model is None:
            model = load_model()
        
        # Dummy prediction if no proper model
        if not TF_AVAILABLE or model == "dummy_model":
            logger.warning("Using dummy prediction")
            return "Tomato - Healthy", 0.95
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # Standard size for most models
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction based on model type
        if isinstance(model, tf.keras.Model):
            # Keras model prediction
            predictions = model.predict(img_array, verbose=0)[0]
        elif isinstance(model, tf.lite.Interpreter):
            # TFLite model prediction
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            model.invoke()
            predictions = model.get_tensor(output_details[0]['index'])[0]
        else:
            raise ValueError("Unsupported model type")
        
        # Process predictions
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        # Ensure we don't go out of bounds
        if predicted_idx < len(CLASS_NAMES):
            disease = CLASS_NAMES[predicted_idx]
        else:
            disease = "Unknown"
            confidence = 0.0
        
        logger.info(f"Prediction: {disease} with confidence {confidence:.2f}")
        return disease, confidence
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Unknown", 0.0