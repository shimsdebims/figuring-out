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

def is_plant_image(image, threshold=0.15):
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
        
        # Broader range for plant material detection (includes yellowing leaves)
        lower_green = np.array([25, 20, 20])  # More inclusive hue range
        upper_green = np.array([95, 255, 255])  # Include yellows and blue-greens
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.mean(mask > 0)
        
        # For debugging
        if st.session_state.get('debug_mode', False):
            st.write(f"Plant material detected: {green_percentage:.2%} (threshold: {threshold:.2%})")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image")
            with col2:
                st.image(mask, caption="Plant Material Mask")
                
        return green_percentage > threshold
    except Exception as e:
        logger.warning(f"Plant verification failed: {str(e)}")
        st.error(f"Error in plant detection: {str(e)}")
        return False

class MockModel:
    """Fallback mock model with realistic predictions"""
    def predict(self, x):
        # Return more diverse predictions as a fallback
        preds = np.zeros((1, len(CLASS_NAMES)))
        # Generate a somewhat random prediction but biased toward healthy
        class_idx = min(len(CLASS_NAMES) - 1, int(np.random.random() * 3))  # Bias toward first few classes
        confidence = 0.7 + (np.random.random() * 0.2)  # Random confidence between 70-90%
        preds[0][class_idx] = confidence
        return preds

def verify_model_file(file_path):
    """Verify the model file is valid"""
    if not os.path.exists(file_path):
        return False, "Model file not found"
        
    file_size = os.path.getsize(file_path)
    if file_size < 10_000_000:  # TFLite models should be at least 10MB
        return False, f"Model file too small ({file_size} bytes)"
        
    try:
        # Attempt to load the model
        interpreter = tf.lite.Interpreter(model_path=file_path)
        interpreter.allocate_tensors()
        return True, "Model verified successfully"
    except Exception as e:
        return False, f"Model validation failed: {str(e)}"

@st.cache_resource
def load_model():
    """Robust model loader with multiple fallbacks"""
    # Direct download URL (no need for ID extraction)
    TFLITE_URL = "https://drive.google.com/uc?export=download&id=1p-wYYieER16UuSmP4fdEwEpnqzUxXaGi"
    TFLITE_PATH = "Model/plant_disease_model.tflite"
    
    # Ensure Model directory exists
    os.makedirs("Model", exist_ok=True)
    
    try:
        # 1. Check local cache first
        if os.path.exists(TFLITE_PATH):
            is_valid, message = verify_model_file(TFLITE_PATH)
            if is_valid:
                st.toast("Loading model from cache", icon="✅")
                interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
                interpreter.allocate_tensors()
                
                # Verify model has expected input/output structure
                input_details = interpreter.get_input_details()
                if len(input_details) == 0:
                    raise Exception("Model has no input tensors")
                    
                # Basic model diagnostics
                if st.session_state.get('debug_mode', False):
                    st.write("Model loaded successfully")
                    st.write(f"Input details: {input_details}")
                    st.write(f"Output details: {interpreter.get_output_details()}")
                
                return interpreter
            else:
                st.warning(f"Cached model invalid: {message}, will re-download")
                # Delete the invalid file
                os.remove(TFLITE_PATH)

        # 3. Download with multiple fallback methods
        with st.spinner("🌱 Downloading disease detector (This may take a few minutes)..."):
            # Method 1: Direct download with requests
            try:
                import requests
                response = requests.get(TFLITE_URL, stream=True, timeout=90)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                downloaded = 0
                
                with open(TFLITE_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress_bar.progress(min(1.0, downloaded / total_size))
                            
                # Verify download
                is_valid, message = verify_model_file(TFLITE_PATH)
                if not is_valid:
                    raise Exception(f"Downloaded model invalid: {message}")
                    
            except Exception as e:
                st.warning(f"Download method 1 failed: {str(e)}")
                
                # Method 2: Alternative URL format
                try:
                    alt_url = "https://drive.google.com/uc?export=download&confirm=yes&id=1p-wYYieER16UuSmP4fdEwEpnqzUxXaGi"
                    import gdown
                    gdown.download(alt_url, TFLITE_PATH, quiet=False)
                    
                    # Verify download
                    is_valid, message = verify_model_file(TFLITE_PATH)
                    if not is_valid:
                        raise Exception(f"Downloaded model invalid: {message}")
                except Exception as e:
                    st.warning(f"Download method 2 failed: {str(e)}")
                    raise Exception("All download methods failed")

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
        start_time = time.time()
        
        # Convert input to PIL Image
        if isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        elif hasattr(image_input, 'read'):
            img = Image.open(image_input)
        else:
            img = Image.open(BytesIO(image_input.getvalue()))
            
        # Preprocess image
        img = img.convert('RGB')
        
        # Debug visualization
        if st.session_state.get('debug_mode', False):
            st.write("### Preprocessing steps")
            st.image(img, caption="Original Image", width=300)
        
        # Resize to model input size
        img_resized = img.resize((224, 224))
        
        if st.session_state.get('debug_mode', False):
            st.image(img_resized, caption="Resized to 224x224", width=224)
        
        # Convert to normalized numpy array
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Get model from session state
        model = st.session_state.model
        
        # Handle TFLite vs Mock model cases
        if isinstance(model, tf.lite.Interpreter):
            # TFLite inference
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            if st.session_state.get('debug_mode', False):
                st.write(f"Input shape: {input_details[0]['shape']}")
                st.write(f"Input data type: {input_details[0]['dtype']}")
            
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
        
        inference_time = time.time() - start_time
        
        if st.session_state.get('debug_mode', False):
            st.write("### Prediction details")
            st.write(f"Inference time: {inference_time:.2f} seconds")
            st.write(f"Raw predictions: {predictions}")
            st.write(f"Predicted class index: {predicted_idx}")
            st.write(f"Class names: {CLASS_NAMES}")
        
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        st.error(f"Error in disease prediction: {str(e)}")
        return "Error", 0.0

def test_model_sample():
    """Run a test prediction on a sample image"""
    try:
        sample_path = "Assets/TomatoSeptoriaLeafSpot(3628).JPG"
        if not os.path.exists(sample_path):
            return "Sample image not found"
        
        with open(sample_path, "rb") as f:
            img_bytes = f.read()
        
        disease, confidence = predict_disease(img_bytes)
        return f"Test prediction: {disease} with {confidence:.2%} confidence"
    except Exception as e:
        return f"Test failed: {str(e)}"