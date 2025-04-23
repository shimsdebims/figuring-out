"""
Plant Disease Prediction Model with TFLite support
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

# Load class names
try:
    with open("disease_info.json") as f:
        disease_info = json.load(f)
        CLASS_NAMES = list(disease_info.keys())
except Exception as e:
    logger.error(f"Error loading disease info: {str(e)}")
    CLASS_NAMES = ["Healthy"]

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

class MockModel:
    """Fallback mock model with realistic predictions"""
    def predict(self, x):
        preds = np.zeros((1, len(CLASS_NAMES)))
        preds[0][0] = 0.87  # 87% confidence for first class
        return preds

@st.cache_resource
def load_model():
    TFLITE_URL = "https://drive.google.com/uc?id=1p-wYYieER16UuSmP4fdEwEpnqzUxXaGi"
    TFLITE_PATH = "Model/plant_disease_model.tflite"
    
    try:
        # 1. Check local cache first
        if os.path.exists(TFLITE_PATH):
            st.toast("Loading model from cache", icon="✅")
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            return interpreter
            
        # 2. Download from Google Drive
        os.makedirs("Model", exist_ok=True)
        with st.spinner("Downloading AI model (80MB - first time only)..."):
            import gdown
            from urllib.parse import urlparse
            import shutil
            
            # Temporary download path
            temp_path = "Model/temp_model.tflite"
            
            # Download with progress
            try:
                gdown.download(TFLITE_URL, temp_path, quiet=False)
                
                # Verify download completed
                if not os.path.exists(temp_path):
                    raise Exception("Download failed - no file received")
                    
                # Verify file size (should be >50MB)
                if os.path.getsize(temp_path) < 50_000_000:
                    raise Exception(f"File too small ({os.path.getsize(temp_path)} bytes)")
                    
                # Move to final location
                shutil.move(temp_path, TFLITE_PATH)
                
            except Exception as download_error:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise download_error
                
        # 3. Load the downloaded model
        if os.path.exists(TFLITE_PATH):
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            st.toast("Model loaded successfully!", icon="✅")
            return interpreter
            
        raise Exception("Model file missing after download")
        
    except Exception as e:
        st.error(f"""
        ⚠️ Model loading failed: {str(e)}
        Using demo mode with limited functionality
        """)
        return MockModel()

def predict_disease(image_input):
    """Robust prediction function for both TFLite and mock models"""
    try:
        # Convert input to PIL Image
        if isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        elif hasattr(image_input, 'read'):
            img = Image.open(image_input)
        else:
            img = Image.open(BytesIO(image_input.getvalue()))
            
        # Preprocess image
        img = img.convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Get model from session state
        model = st.session_state.model
        
        # Handle TFLite vs Mock model cases
        if isinstance(model, tf.lite.Interpreter):
            # TFLite inference
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            model.set_tensor(input_details[0]['index'], img_array)
            model.invoke()
            predictions = model.get_tensor(output_details[0]['index'])[0]
        else:
            # Mock model fallback
            predictions = model.predict(img_array)[0]
        
        # Process results
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        disease = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
        
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error", 0.0