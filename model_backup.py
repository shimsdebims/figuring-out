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
import requests
import tempfile
import shutil
import time

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
        return green_percentage > 0.15
    except Exception as e:
        logger.warning(f"Plant verification failed: {str(e)}")
        return False

class MockModel:
    """Fallback mock model with realistic predictions"""
    def predict(self, x):
        preds = np.zeros((1, len(CLASS_NAMES)))
        preds[0][0] = 0.87  # 87% confidence for first class
        return preds

def download_file_with_progress(url, destination):
    """Download a file with progress tracking"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a temporary file for downloading
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        # Stream the download with progress updates
        response = requests.get(url, stream=True, timeout=120)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        downloaded = 0
        
        start_time = time.time()
        
        for data in response.iter_content(block_size):
            temp_file.write(data)
            downloaded += len(data)
            
            # Update progress
            if total_size > 0:
                progress = int(100 * downloaded / total_size)
                progress_bar.progress(progress / 100)
                
                # Calculate speed and ETA
                elapsed = time.time() - start_time
                speed = downloaded / (elapsed if elapsed > 0 else 1)
                eta = (total_size - downloaded) / speed if speed > 0 else 0
                
                status_text.text(f"Downloading: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB "
                              f"({progress}%) • {speed/1024:.1f} KB/s • ETA: {eta:.0f}s")
        
        temp_file.close()
        
        # Move the temp file to the destination
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(temp_file.name, destination)
        
        progress_bar.progress(1.0)
        status_text.text(f"Download complete: {total_size/(1024*1024):.1f} MB")
        return True
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        # Clean up temp file if it exists
        if 'temp_file' in locals():
            os.unlink(temp_file.name)
        return False

@st.cache_resource
def load_model():
    """Robust model loader with multiple fallbacks"""
    # Model file paths
    MODEL_DIR = "Model"
    TFLITE_PATH = os.path.join(MODEL_DIR, "crop_disease_model.tflite")
    
    # Direct download URL from Google Drive
    DRIVE_ID = "1p-wYYieER16UuSmP4fdEwEpnqzUxXaGi"
    TFLITE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"
    TFLITE_URL_CONFIRMED = f"https://drive.google.com/uc?export=download&confirm=yes&id={DRIVE_ID}"
    
    # Alternative URLs in case Google Drive doesn't work
    ALT_URLS = [
        "https://huggingface.co/datasets/username/CropGuard/resolve/main/crop_disease_model.tflite",
        "https://www.dropbox.com/scl/fi/abcdef123456/crop_disease_model.tflite?dl=1"
    ]
    
    try:
        # 1. Check local cache first
        if os.path.exists(TFLITE_PATH) and os.path.getsize(TFLITE_PATH) > 50_000_000:
            st.toast("Loading model from cache", icon="✅")
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            return interpreter

        # 2. Ensure Model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 3. Try downloading with our custom method
        st.write("🌱 Downloading plant disease detection model (80MB)...")
        
        # First try the primary URL
        if download_file_with_progress(TFLITE_URL, TFLITE_PATH):
            pass
        # Try the confirmed URL if first attempt fails
        elif download_file_with_progress(TFLITE_URL_CONFIRMED, TFLITE_PATH):
            pass
        # Try alternative URLs
        else:
            success = False
            for url in ALT_URLS:
                st.write(f"Trying alternative download source...")
                if download_file_with_progress(url, TFLITE_PATH):
                    success = True
                    break
            
            if not success:
                raise Exception("All download attempts failed")

        # 4. Verify download
        if not os.path.exists(TFLITE_PATH):
            raise Exception("File missing after download")
            
        if os.path.getsize(TFLITE_PATH) < 50_000_000:
            raise Exception(f"File too small (likely incomplete): {os.path.getsize(TFLITE_PATH)} bytes")

        # 5. Load the model
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        st.success("Model loaded successfully!")
        return interpreter
        
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