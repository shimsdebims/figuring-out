import streamlit as st
import datetime
from PIL import Image
import numpy as np
import os
import json
import time
import logging
from io import BytesIO
import sys
from pathlib import Path

# Local imports
from auth import register_user, login_user, is_valid_email
from database import initialize_db, insert_upload, get_user_uploads, insert_feedback
from model import predict_disease, load_model, is_plant_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if styles.css exists
if os.path.exists("styles.css"):
    # Load custom CSS
    def load_css():
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    load_css()
else:
    logger.warning("styles.css not found - app will use default styling")

# Check if necessary directories exist
if not os.path.exists("assets"):
    os.makedirs("assets", exist_ok=True)
    logger.warning("assets directory created - example images may not be available")

# Initialize database
initialize_db()

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.active_tab = "login"
    st.session_state.camera_on = False
    st.session_state.model = None
    st.session_state.model_loaded = False

# Helper functions
def display_disease_info(disease):
    """Display disease information in a styled card"""
    try:
        with open("disease_info.json", "r") as f:
            disease_info = json.load(f)
            
        if disease in disease_info:
            info = disease_info[disease]
            st.markdown(f"""
            <div class="disease-card">
                <h3>{disease}</h3>
                <p><strong>🌿 Symptoms:</strong> {info['symptoms']}</p>
                <p><strong>💊 Treatment:</strong> {info['treatment']}</p>
                <p><strong>🔍 Fun Fact:</strong> {info['fun_fact']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No information found for {disease}")
    except Exception as e:
        st.warning(f"Could not load disease information: {str(e)}")

def log_prediction(user_id, image_path, prediction, confidence):
    """Log prediction details"""
    logger.info(f"User {user_id} prediction: {prediction} (Confidence: {confidence:.2%})")

# Main App
st.title("🌱 Crop Disease Detection")

# Try to load model (only once)
if not st.session_state.get('model_loaded'):
    try:
        with st.spinner("🌱 Loading plant disease model..."):
            # Debugging info
            current_dir = Path(__file__).parent
            st.info(f"Current directory: {current_dir}")
            st.info(f"Model path: {current_dir / 'Model' / 'crop_model.h5'}")
            
            # Load model
            st.session_state.model = load_model()
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
            
    except Exception as e:
        st.error(f"❌ Critical error loading model: {str(e)}")
        st.error("Possible solutions:")
        st.error("1. Verify the model file is not corrupted")
        st.error("2. Check TensorFlow version compatibility")
        st.error("3. Try re-uploading the model file")
        
        # Show more debug info
        with st.expander("Technical details"):
            st.write(f"Python version: {sys.version}")
            st.write(f"TensorFlow version: {tf.__version__}")
            st.write(f"Model file size: {os.path.getsize('Model/crop_model.h5')} bytes")
            
        st.stop()  # Stop the app if model can't load

# Home page (before login)
if not st.session_state.logged_in:
    st.markdown("""
    ## Welcome to CropGuard!
    Detect plant diseases instantly using AI. Upload or capture images of plant leaves to get started.
    """)
    
    # Example images - check if they exist first
    st.subheader("📸 Example Images")
    col1, col2, col3 = st.columns(3)
    example_paths = {
        "assets/example_healthy.jpg": "Healthy Tomato Leaf",
        "assets/example_diseased.jpg": "Diseased Potato Leaf",
        "assets/example_leaf.jpg": "Proper Leaf Close-up"
    }
    
    columns = [col1, col2, col3]
    for i, (path, caption) in enumerate(example_paths.items()):
        with columns[i]:
            if os.path.exists(path):
                st.image(path, caption=caption)
            else:
                st.info(f"Example image not found: {caption}")
    
    st.markdown("""
    ### 📝 Upload Guidelines:
    1. Capture clear, well-lit photos of leaves
    2. Focus on the most affected areas
    3. Avoid blurry or distant shots
    4. Take photos against a neutral background
    """)
    
    # Show login/register in sidebar
    with st.sidebar:
        st.title("User Account")
        st.write("Please login or register to use the detection features")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            if st.button("Login"):
                if login_username and login_password:
                    with st.spinner("Authenticating..."):
                        success, message = login_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.session_state.user_id = message
                            st.success("Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
        
        with tab2:
            reg_username = st.text_input("New Username")
            reg_password = st.text_input("New Password", type="password")
            reg_email = st.text_input("Email")
            
            if st.button("Register"):
                if reg_username and reg_password and reg_email:
                    if not is_valid_email(reg_email):
                        st.error("Please enter a valid email address")
                    else:
                        with st.spinner("Creating account..."):
                            success, message = register_user(reg_username, reg_password, reg_email)
                            if success:
                                st.success("Account created! Please login.")
                            else:
                                st.error(message)

# Main app after login
else:
    with st.sidebar:
        st.title(f"👋 Welcome, {st.session_state.username}!")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.success("Logged out successfully!")
            time.sleep(1)
            st.rerun()
        
        st.subheader("📚 Your Recent Uploads")
        uploads = get_user_uploads(st.session_state.user_id, limit=3)
        if uploads:
            for upload in uploads:
                with st.expander(f"{upload['disease_prediction']} ({upload['upload_date'].strftime('%m/%d')})"):
                    if 'image' in upload:
                        st.image(upload['image'], use_column_width=True)
                    st.write(f"Confidence: {upload.get('confidence', 0):.0%}")
                    if 'feedback' in upload:
                        st.write(f"Your feedback: {upload['feedback']}")
        else:
            st.info("No uploads yet")

    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Camera Capture", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Plant Image")
        st.markdown("""
        **Upload guidelines:**
        - Clear photo of a single leaf
        - Well-lit with minimal shadows
        - Focus on affected areas
        """)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your uploaded image", use_column_width=True)
                
                if st.button("Analyze Image"):
                    if not st.session_state.model_loaded:
                        st.error("❌ Model not loaded. Cannot analyze images.")
                    else:
                        with st.spinner("🔍 Analyzing plant health..."):
                            # Verify it's a plant image
                            if not is_plant_image(image):
                                st.error("This doesn't appear to be a plant leaf image. Please upload a clear photo of a plant leaf.")
                            else:
                                # Save to memory
                                img_bytes = uploaded_file.getvalue()
                                
                                # Make prediction
                                disease, confidence = predict_disease(BytesIO(img_bytes))
                                log_prediction(st.session_state.user_id, "upload", disease, confidence)
                                
                                # Save to database
                                upload_data = {
                                    "user_id": st.session_state.user_id,
                                    "image": img_bytes,
                                    "disease_prediction": disease,
                                    "confidence": confidence,
                                    "upload_date": datetime.datetime.now(datetime.UTC)
                                }
                                insert_upload(upload_data)
                                
                                st.success(f"🔬 Detection Result: **{disease}**")
                                st.metric("Confidence Level", f"{confidence:.0%}")
                                display_disease_info(disease)
                                
                                # Feedback system
                                feedback = st.radio(
                                    "Was this prediction accurate?",
                                    ["Select option", "👍 Accurate", "👎 Inaccurate"],
                                    key="feedback"
                                )
                                if feedback != "Select option":
                                    insert_feedback({
                                        "user_id": st.session_state.user_id,
                                        "prediction": disease,
                                        "feedback": feedback,
                                        "timestamp": datetime.datetime.now(datetime.UTC)
                                    })
                                    st.success("Thank you for your feedback!")
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                logger.error(f"Upload error: {str(e)}")
    
    with tab2:
        st.header("Real-time Camera Capture")
        st.markdown("Capture clear photos of plant leaves for instant analysis")
        
        if st.button(f"{'📷 Turn Off Camera' if st.session_state.camera_on else '📸 Turn On Camera'}"):
            st.session_state.camera_on = not st.session_state.camera_on
            st.rerun()
        
        if st.session_state.camera_on:
            captured_image = st.camera_input("Point camera at a plant leaf")
            
            if captured_image is not None:
                if st.button("Analyze Capture"):
                    if not st.session_state.model_loaded:
                        st.error("❌ Model not loaded. Cannot analyze images.")
                    else:
                        with st.spinner("🔍 Analyzing plant health..."):
                            try:
                                image = Image.open(captured_image)
                                
                                if not is_plant_image(image):
                                    st.error("This doesn't appear to be a plant leaf. Please capture a clear photo of a plant leaf.")
                                else:
                                    # Save to memory
                                    img_bytes = captured_image.getvalue()
                                    
                                    # Make prediction
                                    disease, confidence = predict_disease(BytesIO(img_bytes))
                                    log_prediction(st.session_state.user_id, "camera", disease, confidence)
                                    
                                    # Save to database
                                    upload_data = {
                                        "user_id": st.session_state.user_id,
                                        "image": img_bytes,
                                        "disease_prediction": disease,
                                        "confidence": confidence,
                                        "upload_date": datetime.datetime.now(datetime.UTC)
                                    }
                                    insert_upload(upload_data)
                                    
                                    st.success(f"🔬 Detection Result: **{disease}**")
                                    st.metric("Confidence Level", f"{confidence:.0%}")
                                    display_disease_info(disease)
                                    
                                    # Feedback system
                                    feedback = st.radio(
                                        "Was this prediction accurate?",
                                        ["Select option", "👍 Accurate", "👎 Inaccurate"],
                                        key="camera_feedback"
                                    )
                                    if feedback != "Select option":
                                        insert_feedback({
                                            "user_id": st.session_state.user_id,
                                            "prediction": disease,
                                            "feedback": feedback,
                                            "timestamp": datetime.datetime.now(datetime.UTC)
                                        })
                                        st.success("Thank you for your feedback!")
                            
                            except Exception as e:
                                st.error(f"❌ Error processing image: {str(e)}")
                                logger.error(f"Camera error: {str(e)}")
    
    with tab3:
        st.header("About CropGuard")
        st.markdown("""
        ### 🌿 Plant Disease Detection System
        Uses AI to identify diseases in crops like tomatoes, potatoes, corn, and rice.
        
        **How it works:**
        1. Capture/upload leaf images
        2. AI analyzes the image
        3. Get instant diagnosis and treatment advice
        
        **For best results:**
        - Use clear, well-lit photos
        - Focus on affected leaves
        - Avoid blurry or distant shots
        """)