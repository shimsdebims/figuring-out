import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# Define the class names based on disease_info.json
try:
    with open("disease_info.json", "r") as f:
        disease_info = json.load(f)
        CLASS_NAMES = list(disease_info.keys())
except Exception as e:
    print(f"Error loading disease_info.json: {str(e)}")
    # Default class names if file not found
    CLASS_NAMES = [
        "Tomato - Healthy", "Tomato - Leaf Mold", "Tomato - Yellow Leaf Curl Virus", 
        "Tomato - Septoria Leaf Spot", "Potato - Healthy", "Potato - Late Blight",
        "Potato - Early Blight", "Potato - Scab", "Corn - Healthy", "Corn - Northern Leaf Blight",
        "Corn - Common Rust", "Corn - Gray Leaf Spot", "Rice - Healthy", "Rice - Blast",
        "Rice - Bacterial Leaf Blight", "Rice - Brown Spot"
    ]

# Load the model
def load_model():
    """Load the crop disease detection model"""
    # Try different model paths
    model_paths = [
        "Model/best_model.h5",
        "Model/plant_disease_model.h5",
        "Model/model.tflite"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Found model at {model_path}")
            # Check file extension to determine model type
            if model_path.endswith('.h5'):
                try:
                    # Add this for compatibility with newer TF versions
                    tf.keras.utils.disable_interactive_logging()
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print(f"Successfully loaded Keras model from {model_path}")
                    return model
                except Exception as e:
                    print(f"Failed to load Keras model from {model_path}: {str(e)}")
            
            elif model_path.endswith('.tflite'):
                try:
                    # Load TFLite model
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                    print(f"Successfully loaded TFLite model from {model_path}")
                    return interpreter  # Return TFLite interpreter directly
                except Exception as e:
                    print(f"Failed to load TFLite model from {model_path}: {str(e)}")
    
    raise FileNotFoundError(f"No valid model found in the Model directory. Please ensure your trained model is in the Model directory.")

def predict_disease(image_path, model=None):
    """
    Predict plant disease from an image
    
    Args:
        image_path: Path to the image file
        model: Optional pre-loaded model (to avoid loading for each prediction)
        
    Returns:
        disease: Predicted disease name
        confidence: Confidence score (0-1)
    """
    try:
        # Load model if not provided
        if model is None:
            model = load_model()
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # Resize to match model input size
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Check if model is TFLite interpreter
        if isinstance(model, tf.lite.Interpreter):
            # Get input and output tensors
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Set input tensor
            model.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
            
            # Run inference
            model.invoke()
            
            # Get output tensor
            predictions = model.get_tensor(output_details[0]['index'])[0]
        else:
            # Regular Keras model prediction
            predictions = model.predict(img_array, verbose=0)[0]
        
        # Get prediction and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        # Ensure predicted_class_idx is within bounds of CLASS_NAMES
        if predicted_class_idx < len(CLASS_NAMES):
            disease = CLASS_NAMES[predicted_class_idx]
        else:
            disease = "Unknown"
        
        return disease, confidence
    
    except Exception as e:
        # Return default values if prediction fails
        print(f"Prediction error: {str(e)}")
        return "Unknown", 0.0