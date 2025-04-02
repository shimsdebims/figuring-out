import streamlit as st
import pickle
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
import datetime
from pathlib import Path
import hashlib
import torch

import torch
import torchvision.models as models
from torchvision import transforms

# Import authentication and database functions
from auth import register_user, login_user
from database import init_db, insert_upload, get_user_uploads

# Initialize the database
init_db()

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Load model
def load_model():
    # 1. Initialize the exact same architecture as used in training
    model = models.resnet50(weights=None)  # No pre-trained weights
    
    # 2. Recreate the custom classifier head (must match training exactly)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 15)  #  classes based on your training code
    )
    
    # 3. Load the saved state_dict
    model.load_state_dict(torch.load("Model/crop_disease_model.pth", map_location='cpu'))
    
    # 4. Set to evaluation mode
    model.eval()
    return model

# Load disease information
def load_disease_info():
    with open('disease_info.json', 'r') as file:
        disease_info = json.load(file)
    return disease_info

# Function to make predictions
def predict_disease(uploaded_file, model):
    # Load disease info
    disease_info = load_disease_info()
    
    # Make prediction
    disease, confidence, fun_fact, treatment = predict_disease_streamlit(
        uploaded_file, model, disease_info
    )
    
    return disease, confidence, fun_fact, treatment
    # For demonstration purposes:
    image = Image.open(image)
    image = image.resize((224, 224))
    
    # Convert image to appropriate format for your model
    # image_array = np.array(image) / 255.0
    # prediction = model.predict(np.expand_dims(image_array, axis=0))
    
    # Simulating a prediction result
    prediction = "Apple Black Rot"
    confidence = 0.92
    
    return prediction, confidence

# Function to save uploaded image
def save_uploaded_image(uploaded_file, username):
    # Create directory if it doesn't exist
    save_dir = Path(f"uploads/{username}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(uploaded_file.name).suffix
    filename = f"{timestamp}{file_extension}"
    
    # Save the file
    save_path = save_dir / filename
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(save_path)

# Main app
def main():
    st.title("Crop Disease Detection")
    
    # Sidebar for authentication
    with st.sidebar:
        st.title("User Account")
        
        if not st.session_state.logged_in:
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                st.subheader("Login")
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("Login"):
                    if login_username and login_password:
                        success, user_id = login_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.session_state.user_id = user_id
                            st.success("Successfully logged in!")
                            st.rerun()
                        else:
                            st.error(user_id)  # Error message
                    else:
                        st.warning("Please enter both username and password")
            
            with tab2:
                st.subheader("Register")
                reg_username = st.text_input("Username", key="reg_username")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_email = st.text_input("Email", key="reg_email")
                
                if st.button("Register"):
                    if reg_username and reg_password and reg_email:
                        success, message = register_user(reg_username, reg_password, reg_email)
                        if success:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(message)
                    else:
                        st.warning("Please fill all fields")
        
        else:
            st.write(f"Logged in as: **{st.session_state.username}**")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.user_id = None
                st.rerun()
            
            st.subheader("Your Previous Uploads")
            if st.session_state.user_id:
                user_uploads = get_user_uploads(st.session_state.user_id)
                if user_uploads:
                    for upload in user_uploads:
                        with st.expander(f"{upload['disease_prediction']} - {upload['upload_date']}"):
                            st.write(f"Confidence: {upload['confidence']:.2f}")
                            # Uncomment to display images
                            # if os.path.exists(upload['image_path']):
                            #     st.image(upload['image_path'], width=150)
                else:
                    st.info("You haven't uploaded any images yet.")
    
    # Main content
    if st.session_state.logged_in:
        st.subheader("Upload an image of your crop")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                st.image(uploaded_file, caption="Uploaded Image", width=300)
                
                # Load model and disease info
                model = load_model()  # Comment this if you don't have the model yet
                disease_info = load_disease_info()
                
                if st.button("Predict Disease"):
                    # Save the uploaded image
                    image_path = save_uploaded_image(uploaded_file, st.session_state.username)
                    
                    # Make prediction
                    disease, confidence, fun_fact, treatment = predict_disease(uploaded_file, model)

                    
                    # Save to database
                    upload_data = {
                        "user_id": st.session_state.user_id,
                        "image_path": image_path,
                        "disease_prediction": disease,
                        "confidence": confidence,
                        "upload_date": datetime.datetime.utcnow()
                    }
                    insert_upload(upload_data)
                    
                    # Display results
                    st.success(f"Prediction: {disease}")
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2f}")
                    
                    # Display disease information
                    if disease in disease_info:
                        info = disease_info[disease]
                        st.subheader("Disease Information")
                        st.write(f"**Fun Fact:** {info['fun_fact']}")
                        st.subheader("Treatment Method")
                        st.write(info['treatment'])
                    else:
                        st.warning("Detailed information not available for this disease")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    else:
        st.info("Please login or register to use the crop disease detection tool")

if __name__ == "__main__":
    main()