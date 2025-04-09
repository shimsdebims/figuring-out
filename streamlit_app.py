import streamlit as st
import datetime
from PIL import Image
import numpy as np
import os
import json
import cv2
from io import BytesIO
import time

# Import local modules
from auth import register_user, login_user
from database import initialize_db, insert_upload, get_user_uploads
from model import predict_disease, load_model

# Load custom CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize database
initialize_db()

def display_disease_info(disease):
    """Display information about the detected disease in a styled card"""
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

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "login"
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'model' not in st.session_state:
    try:
        with st.spinner("🌱 Loading plant disease model..."):
            st.session_state.model = load_model()
            st.session_state.model_loaded = True
            st.session_state.model_load_error = None
            st.toast("Model loaded successfully!", icon="✅")
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_load_error = str(e)
        st.error(f"Model loading failed: {str(e)}")

def set_active_tab(tab):
    st.session_state.active_tab = tab

def toggle_camera():
    st.session_state.camera_on = not st.session_state.camera_on

# Main title with emoji
st.title("🌱 Crop Disease Detection")

# Authentication sidebar
with st.sidebar:
    st.title("User Account")
    
    if not st.session_state.logged_in:
        # Custom tab selector
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", key="login_tab_btn", use_container_width=True):
                set_active_tab("login")
        with col2:
            if st.button("Register", key="register_tab_btn", use_container_width=True):
                set_active_tab("register")
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # Show form based on active tab
        if st.session_state.active_tab == "login":
            st.subheader("Login to Your Account")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button", use_container_width=True):
                if login_username and login_password:
                    with st.spinner("Authenticating..."):
                        success, message = login_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.session_state.user_id = message
                            st.success("Logged in successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.warning("Please enter both username and password")
        
        else:  # register tab
            st.subheader("Create New Account")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_email = st.text_input("Email", key="reg_email")
            
            if st.button("Register", key="register_button", use_container_width=True):
                if reg_username and reg_password and reg_email:
                    with st.spinner("Creating account..."):
                        success, message = register_user(reg_username, reg_password, reg_email)
                        if success:
                            st.success("Account created! Please login.")
                            set_active_tab("login")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill all fields")
    
    else:
        st.write(f"👋 Welcome back, {st.session_state.username}!")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.camera_on = False
            st.success("Logged out successfully!")
            time.sleep(1)
            st.rerun()
        
        st.subheader("📚 Your Recent Uploads")
        uploads = get_user_uploads(st.session_state.user_id)
        if uploads:
            for upload in uploads[-5:]:  # Show last 5 uploads
                with st.expander(f"{upload['disease_prediction']} ({upload['upload_date'].strftime('%Y-%m-%d %H:%M')})"):
                    st.image(upload['image_path'], use_column_width=True)
                    if 'confidence' in upload:
                        st.progress(float(upload['confidence']))
                        st.write(f"Confidence: {upload['confidence']:.0%}")
        else:
            st.info("No uploads yet. Analyze some plants to get started!")

# Main app content
if st.session_state.logged_in:
    # Model status indicator
    if st.session_state.model_load_error:
        st.warning(f"⚠️ Model loading issue: {st.session_state.model_load_error}")
        st.info("The app will use reduced functionality with sample predictions")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Camera Capture", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Plant Image")
        st.markdown("Upload a clear photo of a plant leaf to detect potential diseases.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your uploaded image", use_column_width=True)
                
                if st.button("Analyze Image", use_container_width=True):
                    with st.spinner("🔍 Analyzing plant health..."):
                        # Ensure upload directory exists
                        upload_dir = f"uploads/{st.session_state.username}"
                        os.makedirs(upload_dir, exist_ok=True)
                        
                        # Save the image file
                        image_path = os.path.join(upload_dir, uploaded_file.name)
                        with open(image_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Make prediction using the model
                        if st.session_state.model_loaded:
                            disease, confidence = predict_disease(image_path, st.session_state.model)
                        else:
                            st.warning("Model not loaded. Using sample prediction.")
                            disease = "Unknown"
                            confidence = 0.0
                        
                        # Save to database
                        upload_data = {
                            "user_id": st.session_state.user_id,
                            "image_path": image_path,
                            "disease_prediction": disease,
                            "confidence": confidence,
                            "upload_date": datetime.datetime.now(datetime.UTC)
                        }
                        insert_upload(upload_data)
                        
                        st.success(f"🔬 Detection Result: **{disease}**")
                        st.metric("Confidence Level", f"{confidence:.0%}")
                        
                        # Display disease information
                        display_disease_info(disease)
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
    
    with tab2:
        st.header("Real-time Camera Capture")
        st.markdown("Use your device's camera to capture and analyze plant leaves in real-time.")
        
        if st.button(f"{'📷 Turn Off Camera' if st.session_state.camera_on else '📸 Turn On Camera'}", 
                    use_container_width=True):
            toggle_camera()
        
        if st.session_state.camera_on:
            # Use Streamlit's camera input
            captured_image = st.camera_input("Point camera at a plant leaf", label_visibility="collapsed")
            
            if captured_image is not None:
                if st.button("Analyze Capture", use_container_width=True):
                    with st.spinner("🔍 Analyzing plant health..."):
                        try:
                            # Ensure upload directory exists
                            upload_dir = f"uploads/{st.session_state.username}"
                            os.makedirs(upload_dir, exist_ok=True)
                            
                            # Save the image file
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_path = os.path.join(upload_dir, f"capture_{timestamp}.jpg")
                            
                            # Convert BytesIO to Image and save
                            image = Image.open(captured_image)
                            image.save(image_path)
                            
                            # Make prediction using the model
                            if st.session_state.model_loaded:
                                disease, confidence = predict_disease(image_path, st.session_state.model)
                            else:
                                st.warning("Model not loaded. Using sample prediction.")
                                disease = "Unknown"
                                confidence = 0.0
                            
                            # Save to database
                            upload_data = {
                                "user_id": st.session_state.user_id,
                                "image_path": image_path,
                                "disease_prediction": disease,
                                "confidence": confidence,
                                "upload_date": datetime.datetime.now(datetime.UTC)
                            }
                            insert_upload(upload_data)
                            
                            st.success(f"🔬 Detection Result: **{disease}**")
                            st.metric("Confidence Level", f"{confidence:.0%}")
                            
                            # Display disease information
                            display_disease_info(disease)
                        
                        except Exception as e:
                            st.error(f"❌ Error processing captured image: {str(e)}")
        else:
            st.info("Turn on the camera to capture and analyze plant leaves")
    
    with tab3:
        st.header("About This App")
        st.markdown("""
        ### 🌱 Crop Disease Detection System
        
        This application helps farmers and gardeners identify potential diseases in their crops 
        using machine learning technology.
        
        **How it works:**
        1. Upload or capture an image of a plant leaf
        2. Our AI model analyzes the image
        3. Get instant diagnosis and treatment recommendations
        
        **Features:**
        - 📤 Image upload for analysis
        - 📷 Real-time camera capture
        - 🔍 Detailed disease information
        - 📊 History of your previous scans
        
        **Supported Plants:**
        - Tomato
        - Potato
        - Corn
        - Rice
        
        **For best results:**
        - Use clear, well-lit photos
        - Focus on affected leaves
        - Capture both sides of leaves when possible
        """)

else:
    st.info("👋 Please login or register in the sidebar to access the disease detection features")