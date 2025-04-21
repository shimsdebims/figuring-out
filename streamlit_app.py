import streamlit as st
import datetime
from PIL import Image
import os
import json
from io import BytesIO
from pathlib import Path

# Local imports
from auth import register_user, login_user
from database import initialize_db, insert_upload, get_user_uploads, insert_feedback
from model import predict_disease, load_model, is_plant_image

# ======================
# APP CONFIGURATION
# ======================
st.set_page_config(
    page_title="CropGuard",
    page_icon="🍃",  # Leaf icon for browser tab
    layout="wide"
)

# Load CSS
def inject_css():
    """Inject custom CSS styles from file"""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_css()

# Initialize database
initialize_db()

# ======================
# SESSION STATE SETUP
# ======================
if 'logged_in' not in st.session_state:
    st.session_state.update({
        'logged_in': False,
        'username': None,
        'user_id': None,
        'model': load_model(),  # Load model once
        'camera_on': False
    })

# ======================
# HELPER FUNCTIONS
# ======================
def display_disease_info(disease):
    """Display disease information in styled card"""
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
        st.error(f"Could not load disease info: {str(e)}")

def process_image(image, source_type):
    """Handle image processing and prediction"""
    if not is_plant_image(image):
        st.error("Please upload a clear photo of a plant leaf.")
        return
        
    with st.spinner("🔍 Analyzing..."):
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
        
        # Feedback
        feedback = st.radio(
            "Was this accurate?",
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

# ======================
# MAIN APP LAYOUT
# ======================
st.title("🍃 CropGuard - Plant Disease Detection")

# Home Page (Before Login)
if not st.session_state.logged_in:
    st.markdown("""
    ## 🌾 Welcome to CropGuard!
    Detect plant diseases instantly using AI.
    """)
    
    # Example images with error handling
    st.subheader("📸 Example Detections")
    cols = st.columns(3)
    examples = {
        "Healthy Tomato": "assets/tomato_healthy.jpg",
        "Diseased Potato": "assets/potato_diseased.jpg", 
        "Healthy Corn": "assets/corn_healthy.jpg"
    }
    
    for col, (name, path) in zip(cols, examples.items()):
        with col:
            try:
                st.image(path, caption=name, use_container_width=True)
            except:
                st.info(f"Example image not available: {name}")

    # New content sections
    with st.expander("📊 Community Insights"):
        st.write("""
        - 82% accurate tomato disease detection
        - 91% potato blight prevention rate  
        - 76% corn rust identification
        """)
    
    st.markdown("""
    ### 🌱 Why Choose CropGuard?
    - Instant disease detection
    - Science-backed treatment plans  
    - Save up to 40% of your crop yield
    """)

    # Auth in sidebar
    with st.sidebar:
        st.title("Account")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login"):
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
            with st.form("register"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                email = st.text_input("Email")
                if st.form_submit_button("Register"):
                    success, message = register_user(new_user, new_pass, email)
                    if success:
                        st.success("Account created! Please login.")
                    else:
                        st.error(message)

# Main App (After Login)
else:
    with st.sidebar:
        st.title(f"👋 {st.session_state.username}")
        
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
                        st.warning("Image unavailable")
        else:
            st.info("No detection history")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📷 Camera", "🌿 Resources"])
    
    with tab1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
        if uploaded_file and st.button("Analyze"):
            process_image(uploaded_file, "upload")
    
    with tab2:
        st.header("Camera Capture")
        if st.button(f"{'📷 Off' if st.session_state.camera_on else '📸 On'}"):
            st.session_state.camera_on = not st.session_state.camera_on
            st.rerun()
        
        if st.session_state.camera_on:
            img = st.camera_input("Point at plant leaf")
            if img and st.button("Analyze"):
                process_image(img, "camera")
    
    with tab3:
        st.header("Plant Care Resources")
        tips = st.columns(3)
        tips[0].info("**Lighting**: Use natural light")
        tips[1].warning("**Angle**: Shoot leaves flat-on")
        tips[2].success("**Focus**: Close-up of affected areas")
        
        st.markdown("""
        ### 🚜 Best Practices
        - Water in the morning
        - Rotate crops annually  
        - Space plants properly
        
        ### 📚 Learn More
        [USDA Plant Health](https://www.aphis.usda.gov)  
        [Extension Services](https://www.extension.org)
        """)