"""
Main Streamlit Application for Crop Disease Detection
Handles user authentication, image uploads, and disease prediction
"""

import streamlit as st
import datetime
from PIL import Image
import os
import json
from io import BytesIO
from pathlib import Path

# Local imports
from auth import register_user, login_user, is_valid_email
from database import initialize_db, insert_upload, get_user_uploads, insert_feedback
from model import predict_disease, load_model, is_plant_image

# --- App Configuration ---
st.set_page_config(
    page_title="CropGuard - Disease Detection",
    page_icon="🍃",  # Leaf icon
    layout="wide"
)

# --- CSS Injection ---
def inject_css():
    """Inject custom CSS styles"""
    css_file = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS file not found!")

inject_css()

# --- Database Initialization ---
initialize_db()

# --- Session State Setup ---
if 'logged_in' not in st.session_state:
    st.session_state.update({
        'logged_in': False,
        'username': None,
        'user_id': None,
        'model': load_model(),  # Load model once
        'camera_on': False
    })

# --- Helper Functions ---
def display_disease_info(disease):
    """Display disease information in a styled card"""
    try:
        with open("disease_info.json") as f:
            disease_info = json.load(f)
            
        if disease in disease_info:
            info = disease_info[disease]
            st.markdown(f"""
            <div class="disease-card">
                <h3>🌱 {disease}</h3>
                <p><strong>🌿 Symptoms:</strong> {info['symptoms']}</p>
                <p><strong>💊 Treatment:</strong> {info['treatment']}</p>
                <p><strong>🔍 Fun Fact:</strong> {info['fun_fact']}</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load disease information: {str(e)}")

def process_image(image, source_type):
    """Handle image processing and prediction"""
    if not is_plant_image(image):
        st.error("This doesn't appear to be a plant leaf image.")
        return
        
    with st.spinner("🔍 Analyzing plant health..."):
        disease, confidence = predict_disease(image)
        
        # Save to database
        upload_data = {
            "user_id": st.session_state.user_id,
            "image": image.getvalue(),
            "disease_prediction": disease,
            "confidence": confidence,
            "upload_date": datetime.datetime.now(datetime.UTC),
            "source": source_type
        }
        insert_upload(upload_data)
        
        # Display results
        st.success(f"🔬 Detection Result: **{disease}**")
        st.metric("Confidence Level", f"{confidence:.0%}")
        display_disease_info(disease)
        
        # Feedback system
        feedback = st.radio(
            "Was this prediction accurate?",
            ["Select option", "👍 Accurate", "👎 Inaccurate"],
            key=f"feedback_{source_type}"
        )
        if feedback != "Select option":
            insert_feedback({
                "user_id": st.session_state.user_id,
                "prediction": disease,
                "feedback": feedback,
                "timestamp": datetime.datetime.now(datetime.UTC)
            })

# --- Main App Layout ---
st.title("🍃 CropGuard - Plant Disease Detection")

# --- Home Page (Before Login) ---
if not st.session_state.logged_in:
    st.markdown("""
    ## 🌾 Welcome to CropGuard!
    Detect plant diseases instantly using AI. Upload or capture images of plant leaves to get started.
    """)
    
    # Example images with fallback
    st.subheader("📸 Example Detection Results")
    cols = st.columns(3)
    examples = [
        ("tomato_healthy.jpg", "Healthy Tomato Leaf"),
        ("potato_diseased.jpg", "Diseased Potato Leaf"),
        ("corn_healthy.jpg", "Healthy Corn Leaf")
    ]
    
    for col, (img, caption) in zip(cols, examples):
        with col:
            try:
                st.image(f"assets/{img}", caption=caption, use_container_width=True)
            except:
                st.info(f"Example image unavailable: {caption}")

    # How It Works section
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        - **Capture** clear photos of plant leaves
        - **Upload** images through our simple interface
        - **Receive** instant diagnosis and treatment advice
        - **Learn** about plant diseases and prevention
        """)

    # Login/Signup in sidebar
    with st.sidebar:
        st.title("Account")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    success, message = login_user(username, password)
                    if success:
                        st.session_state.update({
                            'logged_in': True,
                            'username': username,
                            'user_id': message
                        })
                        st.rerun()
                    else:
                        st.error(message)
        
        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                email = st.text_input("Email")
                if st.form_submit_button("Register"):
                    if not is_valid_email(email):
                        st.error("Please enter a valid email")
                    else:
                        success, message = register_user(new_user, new_pass, email)
                        if success:
                            st.success("Account created! Please login.")
                        else:
                            st.error(message)

# --- Main App (After Login) ---
else:
    with st.sidebar:
        st.title(f"👋 Welcome, {st.session_state.username}!")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        
        st.subheader("📚 Your History")
        uploads = get_user_uploads(st.session_state.user_id, limit=3)
        if uploads:
            for upload in uploads:
                with st.expander(f"{upload['disease_prediction']} ({upload['confidence']:.0%})"):
                    try:
                        st.image(upload['image'], use_container_width=True)
                    except:
                        st.warning("Could not load image")
        else:
            st.info("No detection history yet")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Camera Capture", "🌱 Plant Care Tips"])
    
    with tab1:
        st.header("Upload Plant Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file and st.button("Analyze"):
            process_image(uploaded_file, "upload")
    
    with tab2:
        st.header("Real-time Detection")
        if st.button(f"{'📷 Turn Off Camera' if st.session_state.camera_on else '📸 Turn On Camera'}"):
            st.session_state.camera_on = not st.session_state.camera_on
            st.rerun()
        
        if st.session_state.camera_on:
            img = st.camera_input("Point camera at plant leaf")
            if img and st.button("Analyze Capture"):
                process_image(img, "camera")
    
    with tab3:
        st.header("Plant Care Resources")
        st.markdown("""
        ### 🚜 Best Practices for Healthy Crops
        - **Watering**: Keep soil moist but not waterlogged
        - **Spacing**: Ensure proper plant spacing for air circulation
        - **Rotation**: Rotate crops annually to prevent disease buildup
        
        ### 🛡️ Prevention Tips
        - Regularly inspect plants for early signs
        - Remove infected plants immediately
        - Use disease-resistant varieties when possible
        
        ### 📚 Learning Resources
        - [USDA Plant Health](https://www.aphis.usda.gov)
        - [Extension Services](https://www.extension.org)
        """)