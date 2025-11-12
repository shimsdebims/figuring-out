"""
Enhanced Plant Disease Prediction Model with Hugging Face MobileNetV2
Supports both direct 16-class predictions and 38-class PlantVillage mapping
"""

import os
import numpy as np
from PIL import Image, ImageStat
import logging
import json
import streamlit as st
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load class names from disease_info.json
try:
    with open("disease_info.json") as f:
        disease_info = json.load(f)
        CLASS_NAMES = list(disease_info.keys())
except Exception as e:
    logger.error(f"Error loading disease info: {str(e)}")
    CLASS_NAMES = ["Healthy"]

# No mapping needed - disease_info.json now uses PlantVillage class names directly

def is_plant_image(image):
    """Verify image contains plant material using basic checks"""
    try:
        # Convert various input types to PIL Image
        if isinstance(image, bytes):
            img = Image.open(BytesIO(image))
        elif hasattr(image, 'read'):
            img = Image.open(image)
        else:
            img = Image.open(BytesIO(image.getvalue()))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Simple check: look for greenish colors in the image
        # Calculate average color values
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean
        
        # If green channel is dominant and image isn't too dark/bright, likely a plant
        # This is a simple heuristic - plants tend to have more green
        is_greenish = g > r * 0.9 and g > b * 0.9
        is_not_too_dark = (r + g + b) / 3 > 30
        is_not_too_bright = (r + g + b) / 3 < 240
        
        return is_greenish and is_not_too_dark and is_not_too_bright
    except Exception as e:
        logger.warning(f"Plant verification failed: {str(e)}")
        return True  # Allow image through if verification fails

class MockModel:
    """Fallback mock model for testing without actual model"""
    def __init__(self):
        self.classes = CLASS_NAMES
        
    def predict(self, image):
        """Return mock prediction"""
        # Randomly pick a class but favor first few
        probs = np.random.dirichlet(np.ones(len(self.classes)) * 0.3)
        return self.classes[np.argmax(probs)], float(np.max(probs))

@st.cache_resource
def load_model():
    """
    Load MobileNetV2 model from Hugging Face using Transformers
    """
    try:
        st.write("🌱 Loading MobileNetV2 model from Hugging Face...")
        
        from transformers import AutoImageProcessor, TFAutoModelForImageClassification
        
        # Load the model from Hugging Face with TensorFlow
        # Using a public PlantVillage MobileNetV2 model
        model_name = "linkanjarad/mobilenet_v2_1.0_224_plant_disease"
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = TFAutoModelForImageClassification.from_pretrained(
            model_name,
            from_pt=True  # Convert from PyTorch if needed
        )
        
        st.success("✅ Model loaded successfully!")
        return {
            'model': model,
            'processor': processor,
            'framework': 'transformers_tf',
            'classes': None
        }
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        
        # Fallback to mock model
        st.warning("⚠️ Using demo mode - model loading failed. Please contact support.")
        return {'model': MockModel(), 'framework': 'mock', 'classes': CLASS_NAMES}

def preprocess_image(image_input):
    """
    Preprocess image for model input
    Returns: numpy array of shape (1, 224, 224, 3) normalized to [0, 1]
    """
    # Convert input to PIL Image
    if isinstance(image_input, bytes):
        img = Image.open(BytesIO(image_input))
    elif hasattr(image_input, 'read'):
        img = Image.open(image_input)
    else:
        img = Image.open(BytesIO(image_input.getvalue()))
    
    # Preprocess
    img = img.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

# No mapping function needed - disease names match PlantVillage format

def predict_disease(image_input):
    """
    Main prediction function
    Returns: (disease_name, confidence)
    """
    try:
        # Preprocess image
        img_array = preprocess_image(image_input)
        
        # Get model from session state
        model_info = st.session_state.model
        model = model_info['model']
        framework = model_info['framework']
        
        # Run inference based on framework
        if framework == 'transformers_tf':
            import tensorflow as tf
            import numpy as np
            
            # Get processor and model
            processor = model_info['processor']
            
            # Convert image input to PIL Image
            if isinstance(image_input, bytes):
                img = Image.open(BytesIO(image_input))
            elif hasattr(image_input, 'read'):
                img = Image.open(image_input)
            else:
                img = Image.open(BytesIO(image_input.getvalue()))
            
            # Preprocess with the model's processor
            inputs = processor(images=img, return_tensors="tf")
            
            # Make prediction
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = tf.nn.softmax(logits, axis=-1)[0]
            
            # Get top prediction
            predicted_idx = int(tf.argmax(predictions).numpy())
            confidence = float(predictions[predicted_idx].numpy())
            
            # Get class name from model's config
            disease = model.config.id2label[predicted_idx]
            
        elif framework == 'mock':
            # Mock model
            disease, confidence = model.predict(image_input)
        
        else:
            raise Exception(f"Unknown framework: {framework}")
        
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

def get_plantvillage_classes():
    """
    Returns the 38 PlantVillage class names in correct order
    This should match the model's output layer order
    """
    return [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
