import streamlit as st
import datetime
from PIL import Image
import numpy as np
import os
import json
import cv2
from io import BytesIO

# Import local modules
from auth import register_user, login_user
from database import initialize_db, insert_upload, get_user_uploads
from model import predict_disease, load_model

# Initialize database
initialize_db()

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
        st.session_state.model = load_model()
        st.session_state.model_loaded = True
    except Exception as e:
        st.error(f"Could not load model: {str(e)}")
        st.session_state.model_loaded = False

def set_active_tab(tab):
    st.session_state.active_tab = tab

def toggle_camera():
    st.session_state.camera_on = not st.session_state.camera_on

# Main title
st.title("Crop Disease Detection")

# Authentication sidebar
with st.sidebar:
    st.title("User Account")
    
    if not st.session_state.logged_in:
        # Custom tab selector
        col1, col2 = st.columns(2)
        with col1:
            login_style = "color: red;" if st.session_state.active_tab == "login" else ""
            if st.button("Login Tab", key="login_tab_btn", help="Switch to login"):
                set_active_tab("login")
                st.rerun()
        
        with col2:
            register_style = "color: red;" if st.session_state.active_tab == "register" else ""
            if st.button("Register Tab", key="register_tab_btn", help="Switch to register"):
                set_active_tab("register")
                st.rerun()
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # Show form based on active tab
        if st.session_state.active_tab == "login":
            st.subheader("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button"):
                if login_username and login_password:
                    success, message = login_user(login_username, login_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.user_id = message
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Both username and password are required")
        
        else:  # register tab
            st.subheader("Register")
            reg_username = st.text_input("New username", key="reg_username")
            reg_password = st.text_input("New password", type="password", key="reg_password")
            reg_email = st.text_input("Email", key="reg_email")
            
            if st.button("Register", key="register_button"):
                if reg_username and reg_password and reg_email:
                    success, message = register_user(reg_username, reg_password, reg_email)
                    if success:
                        st.success("Registered! Please login.")
                        set_active_tab("login")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("All fields are required")
    
    else:
        st.write(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.session_state.camera_on = False
            st.success("Logged out!")
            st.rerun()
        
        st.subheader("Your Uploads")
        uploads = get_user_uploads(st.session_state.user_id)
        if uploads:
            for upload in uploads[-5:]:  # Show last 5 uploads
                st.write(f"{upload['disease_prediction']} ({upload['upload_date'].strftime('%Y-%m-%d')})")
                if 'confidence' in upload:
                    st.write(f"Confidence: {upload['confidence']:.0%}")
                st.markdown("---")
            
            if len(uploads) > 5:
                st.write(f"... and {len(uploads) - 5} more uploads")
        else:
            st.info("No uploads yet")

# Main app content
if st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Upload Image", "Camera Capture"])
    
    with tab1:
        st.header("Upload Plant Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                if st.button("Analyze Uploaded Image"):
                    with st.spinner("Analyzing image..."):
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
                            st.warning("Model not loaded. Using default prediction.")
                            disease = "Unknown"
                            confidence = 0.0
                        
                        # Save to database
                        upload_data = {
                            "user_id": st.session_state.user_id,
                            "image_path": image_path,
                            "disease_prediction": disease,
                            "confidence": confidence,
                            "upload_date": datetime.datetime.utcnow()
                        }
                        insert_upload(upload_data)
                        
                        st.success(f"Detected: {disease}")
                        st.write(f"Confidence: {confidence:.0%}")
                        
                        # Display disease information
                        display_disease_info(disease)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.header("Camera Capture")
        camera_btn = st.button("Toggle Camera", on_click=toggle_camera)
        
        if st.session_state.camera_on:
            camera_placeholder = st.empty()
            capture_btn = st.button("Capture and Analyze")
            
            camera_img = st.camera_input("Take a picture of the plant")
            
            if camera_img is not None and capture_btn:
                with st.spinner("Analyzing captured image..."):
                    # Ensure upload directory exists
                    upload_dir = f"uploads/{st.session_state.username}"
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # Save the image file
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(upload_dir, f"capture_{timestamp}.jpg")
                    
                    # Convert BytesIO to Image and save
                    image = Image.open(camera_img)
                    image.save(image_path)
                    
                    # Make prediction using the model
                    if st.session_state.model_loaded:
                        disease, confidence = predict_disease(image_path, st.session_state.model)
                    else:
                        st.warning("Model not loaded. Using default prediction.")
                        disease = "Unknown"
                        confidence = 0.0
                    
                    # Save to database
                    upload_data = {
                        "user_id": st.session_state.user_id,
                        "image_path": image_path,
                        "disease_prediction": disease,
                        "confidence": confidence,
                        "upload_date": datetime.datetime.utcnow()
                    }
                    insert_upload(upload_data)
                    
                    st.success(f"Detected: {disease}")
                    st.write(f"Confidence: {confidence:.0%}")
                    
                    # Display disease information
                    display_disease_info(disease)
        else:
            st.info("Turn on the camera to capture and analyze plant images in real-time")

else:
    st.info("Please login or register in the sidebar to detect crop diseases")

def display_disease_info(disease):
    """Display information about the detected disease"""
    try:
        with open("disease_info.json", "r") as f:
            disease_info = json.load(f)
            
        if disease in disease_info:
            info = disease_info[disease]
            st.subheader("Disease Information")
            st.write(f"**Symptoms:** {info['symptoms']}")
            st.write(f"**Treatment:** {info['treatment']}")
            st.write(f"**Fun Fact:** {info['fun_fact']}")
        else:
            st.warning(f"No information found for {disease}")
    except Exception as e:
        st.warning(f"Could not load disease information: {str(e)}")