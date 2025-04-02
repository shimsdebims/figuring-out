import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import datetime
from pathlib import Path
import hashlib

# Import authentication functions
from auth import register_user, login_user
from database import insert_upload, get_user_uploads

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Model and class labels
CLASS_LABELS = {
    0: 'Tomato - Healthy',
    1: 'Tomato - Leaf Mold',
    # ... (your other class labels)
}

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_LABELS))
    model.load_state_dict(torch.load('Model/crop_disease_model.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image.convert('RGB')).unsqueeze(0)

def predict_disease(image, model):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()
    return CLASS_LABELS[preds.item()], confidence

def save_uploaded_image(uploaded_file, username):
    save_dir = Path(f"uploads/{username}")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"{timestamp}_{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(save_path)

def main():
    st.title("🌱 Crop Disease Detection")
    
    # Authentication sidebar
    with st.sidebar:
        st.title("User Account")
        
        if not st.session_state.logged_in:
            # Show both login and register tabs
            login_tab, register_tab = st.tabs(["Login", "Register"])
            
            with login_tab:
                st.subheader("Login")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.button("Login"):
                    if username and password:
                        success, user_id = login_user(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_id = user_id
                            st.success("Logged in successfully!")
                            st.rerun()
                        else:
                            st.error(user_id)  # Shows error message
                    else:
                        st.warning("Please enter both username and password")
            
            with register_tab:
                st.subheader("Register")
                new_username = st.text_input("Choose username")
                new_password = st.text_input("Choose password", type="password")
                email = st.text_input("Email")
                if st.button("Create Account"):
                    if new_username and new_password and email:
                        success, message = register_user(new_username, new_password, email)
                        if success:
                            st.success("Account created! Please login.")
                        else:
                            st.error(message)
                    else:
                        st.warning("Please fill all fields")
        
        else:
            st.write(f"Welcome, {st.session_state.username}!")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.user_id = None
                st.success("Logged out successfully!")
                st.rerun()
            
            st.subheader("Your Uploads")
            uploads = get_user_uploads(st.session_state.user_id)
            if uploads:
                for upload in uploads:
                    with st.expander(f"{upload['disease_prediction']}"):
                        st.write(f"Date: {upload['upload_date'].strftime('%Y-%m-%d')}")
                        st.write(f"Confidence: {upload['confidence']:.2f}")
                        if os.path.exists(upload['image_path']):
                            st.image(upload['image_path'], width=150)
            else:
                st.info("No uploads yet")

    # Main content
    if st.session_state.logged_in:
        st.subheader("Upload a plant image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Analyze"):
                    with st.spinner("Analyzing..."):
                        model = load_model()
                        disease, confidence = predict_disease(image, model)
                        
                        # Save to database
                        image_path = save_uploaded_image(uploaded_file, st.session_state.username)
                        upload_data = {
                            "user_id": st.session_state.user_id,
                            "image_path": image_path,
                            "disease_prediction": disease,
                            "confidence": confidence,
                            "upload_date": datetime.datetime.utcnow()
                        }
                        insert_upload(upload_data)
                        
                        st.success(f"Prediction: {disease}")
                        st.write(f"Confidence: {confidence:.2f}")
                        
                        # Show disease info
                        with open('disease_info.json') as f:
                            disease_info = json.load(f)
                        if disease in disease_info:
                            st.subheader("Disease Information")
                            st.write(disease_info[disease]['treatment'])
                            st.write("Fun fact:", disease_info[disease]['fun_fact'])
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    else:
        st.info("Please login or register to use the disease detection tool")

if __name__ == "__main__":
    main()