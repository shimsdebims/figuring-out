"""
Simple test script to verify model loading and prediction
Run this before deploying to Streamlit
"""

import os
import sys
from PIL import Image
import numpy as np

# Test without streamlit
class MockSessionState:
    """Mock streamlit session state for testing"""
    def __init__(self):
        self.model = None

class MockSt:
    """Mock streamlit module for testing"""
    def __init__(self):
        self.session_state = MockSessionState()
    
    def write(self, text):
        print(f"[INFO] {text}")
    
    def toast(self, text, icon=None):
        print(f"[TOAST] {icon} {text}")
    
    def success(self, text):
        print(f"[SUCCESS] {text}")
    
    def warning(self, text):
        print(f"[WARNING] {text}")
    
    def error(self, text):
        print(f"[ERROR] {text}")
    
    def cache_resource(self, func):
        """Mock cache decorator"""
        return func

# Replace streamlit import
sys.modules['streamlit'] = MockSt()
import streamlit as st

# Now import model
from model_v2 import load_model, predict_disease, is_plant_image, get_plantvillage_classes

def test_plantvillage_classes():
    """Test that we have the correct PlantVillage classes"""
    print("\n=== Testing PlantVillage Classes ===")
    classes = get_plantvillage_classes()
    print(f"Total classes: {len(classes)}")
    print(f"Expected: 38")
    print(f"Match: {'✅' if len(classes) == 38 else '❌'}")
    
    # Show some classes
    print("\nSample classes:")
    for i, cls in enumerate(classes[:5]):
        print(f"  {i}: {cls}")
    print("  ...")
    for i, cls in enumerate(classes[-5:], start=len(classes)-5):
        print(f"  {i}: {cls}")

def test_model_loading():
    """Test model loading"""
    print("\n=== Testing Model Loading ===")
    try:
        model_info = load_model()
        st.session_state.model = model_info
        
        print(f"Framework: {model_info['framework']}")
        print(f"Model type: {type(model_info['model'])}")
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

def test_image_validation():
    """Test plant image validation"""
    print("\n=== Testing Image Validation ===")
    
    # Try to load an example image
    example_images = [
        "Assets/PotatoHealthy(2161).JPG",
        "Assets/TomatoSeptoriaLeafSpot(3628).JPG",
        "Assets/CornCommonRust(3279).JPG"
    ]
    
    for img_path in example_images:
        if os.path.exists(img_path):
            print(f"\nTesting: {img_path}")
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                
                is_plant = is_plant_image(img_bytes)
                print(f"  Is plant: {'✅ Yes' if is_plant else '❌ No'}")
                return img_path
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    print("⚠️ No example images found")
    return None

def test_prediction(image_path):
    """Test prediction on a sample image"""
    print("\n=== Testing Prediction ===")
    
    if not image_path or not os.path.exists(image_path):
        print("❌ No valid image to test")
        return
    
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        print(f"Predicting disease for: {image_path}")
        disease, confidence = predict_disease(img_bytes)
        
        print(f"\n🔬 Prediction Result:")
        print(f"  Disease: {disease}")
        print(f"  Confidence: {confidence:.2%}")
        
        if confidence > 0.7:
            print("  Status: ✅ High confidence")
        elif confidence > 0.5:
            print("  Status: ⚠️ Medium confidence")
        else:
            print("  Status: ❌ Low confidence")
            
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("="*60)
    print("CropGuard Model Testing")
    print("="*60)
    
    # Test 1: PlantVillage classes
    test_plantvillage_classes()
    
    # Test 2: Model loading
    if not test_model_loading():
        print("\n⚠️ Model loading failed, but tests can continue with mock model")
    
    # Test 3: Image validation
    sample_image = test_image_validation()
    
    # Test 4: Prediction
    test_prediction(sample_image)
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)

if __name__ == "__main__":
    main()
