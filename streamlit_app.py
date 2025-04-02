import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import datetime
from pathlib import Path
import hashlib
from auth import register_user, login_user
from database import init_db, insert_upload, get_user_uploads
from database import get_database

# Right after your imports
def test_db_connection():
    try:
        db = get_database()
        collections = db.list_collection_names()
        st.sidebar.success(f"Connected to MongoDB! Collections: {collections}")
    except Exception as e:
        st.sidebar.error(f"MongoDB connection failed: {str(e)}")


# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Set page config
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class labels (must match your training)
CLASS_LABELS = {
    0: 'Tomato - Healthy',
    1: 'Tomato - Leaf Mold',
    2: 'Tomato - Yellow Leaf Curl Virus',
    3: 'Tomato - Septoria Leaf Spot',
    4: 'Potato - Healthy',
    5: 'Potato - Late Blight',
    6: 'Potato - Early Blight',
    7: 'Corn - Healthy',
    8: 'Corn - Northern Leaf Blight',
    9: 'Corn - Common Rust',
    10: 'Corn - Gray Leaf Spot',
    11: 'Rice - Healthy',
    12: 'Rice - Blast',
    13: 'Rice - Bacterial Leaf Blight',
    14: 'Rice - Brown Spot'
}

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_classes = 15
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('Model/crop_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load disease info
def load_disease_info():
    with open('disease_info.json', 'r') as f:
        return json.load(f)

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = image.convert('RGB')
    img = transform(img).unsqueeze(0)
    return img

# Prediction function
def predict_disease(image, model):
    img = preprocess_image(image)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()
    return CLASS_LABELS[predicted_class], confidence

# Function to save uploaded image
def save_uploaded_image(uploaded_file, username):
    save_dir = Path(f"uploads/{username}")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(uploaded_file.name).suffix
    filename = f"{timestamp}{file_extension}"
    save_path = save_dir / filename
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(save_path)

# Main app
test_db_connection()
# Initialize the database
if not init_db():
    st.error("Failed to initialize database connection. Some features may not work.")
else:
    st.success("Database connection established successfully")

def main():
    st.title("🌱 Crop Disease Detection")
    
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
                            st.error(user_id)
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
                            st.error(f"Database error during registration")
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
                        with st.expander(f"{upload['disease_prediction']} - {upload['upload_date'].strftime('%Y-%m-%d %H:%M')}"):
                            st.write(f"Confidence: {upload['confidence']:.2f}")
                            if os.path.exists(upload['image_path']):
                                st.image(upload['image_path'], width=150)
                else:
                    st.info("You haven't uploaded any images yet.")
    
    # Main content
    if st.session_state.logged_in:
        st.subheader("Upload an image of your crop")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                if st.button('Analyze'):
                    with st.spinner('Analyzing the image...'):
                        # Load model and disease info
                        model = load_model()
                        disease_info = load_disease_info()
                        
                        # Make prediction
                        prediction, confidence = predict_disease(image, model)
                        info = disease_info.get(prediction, {})
                        
                        # Save the uploaded image
                        image_path = save_uploaded_image(uploaded_file, st.session_state.username)
                        
                        # Save to database
                        upload_data = {
                            "user_id": st.session_state.user_id,
                            "image_path": image_path,
                            "disease_prediction": prediction,
                            "confidence": confidence,
                            "upload_date": datetime.datetime.utcnow()
                        }
                        success, message = insert_upload(upload_data)
                        if not success:
                            st.error(f"Failed to save to database: {message}")
                        
                        
                        # Display results
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("Prediction Result")
                            st.markdown(f"**{prediction}**")
                            st.image(image, use_column_width=True)
                        
                        with col2:
                            st.subheader("Disease Information")
                            
                            with st.expander("Symptoms", expanded=True):
                                st.info(info.get('symptoms', 'No symptoms information available'))
                            
                            with st.expander("Treatment Recommendations"):
                                st.warning(info.get('treatment', 'No treatment information available'))
                            
                            with st.expander("Did You Know?"):
                                st.success(info.get('fun_fact', 'No fun facts available'))
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    else:
        st.info("Please login or register to use the crop disease detection tool")

if __name__ == "__main__":
    main()