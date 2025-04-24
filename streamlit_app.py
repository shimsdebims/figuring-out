import streamlit as st
import datetime
from PIL import Image
import os
import json
from io import BytesIO
from pathlib import Path
from auth import register_user, login_user, hash_password, is_strong_password
from database import (
    initialize_db, 
    insert_upload, 
    get_user_uploads, 
    insert_feedback, 
    delete_user_account, 
    clear_user_uploads, 
    update_user_password,
    find_user_by_username  
)
import time
from model import predict_disease, load_model, is_plant_image

# ================
# APP CONFIG
# ================
st.set_page_config(
    page_title="CropGuard",
    page_icon="🍃",
    layout="wide"
)

# Display a notice if the model isn't available
if not os.path.exists("Model/plant_disease_model.h5"):
    st.warning("⚠️ Running in demo mode: Full model not available. For demonstration purposes only.")

# Load CSS
def inject_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_css()

# Initialize DB
initialize_db()

# ================
# SESSION STATE
# ================
if 'logged_in' not in st.session_state:
    st.session_state.update({
        'logged_in': False,
        'username': None,
        'user_id': None,
        'camera_on': False,
        'current_tab': "Home"
    })

if 'model' not in st.session_state:
    with st.spinner("🌱 Loading disease detection model..."):
        st.session_state.model = load_model()

# ================
# CORE FUNCTIONS
# ================
def display_disease_info(disease):
    """Display disease information card"""
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
    try:
        if not is_plant_image(image):
            st.error("❌ Please upload a clear photo of a plant leaf")
            return
            
        with st.spinner("🔍 Analyzing..."):
            disease, confidence = predict_disease(image)
            
            if disease == "Error":
                st.error("Analysis failed. Please try another image.")
                return
                
            st.success(f"🔬 Detection Result: **{disease}**")
            st.metric("Confidence Level", f"{confidence:.0%}")
            display_disease_info(disease)
            
            # Save result to database if logged in
            if st.session_state.logged_in:
                try:
                    # Convert image to binary for storage
                    img_bytes = BytesIO()
                    if hasattr(image, 'getvalue'):
                        img_bytes.write(image.getvalue())
                    else:
                        img = Image.open(image)
                        img.save(img_bytes, format='JPEG')
                    
                    # Insert into database
                    insert_upload({
                        "user_id": st.session_state.user_id,
                        "image_binary": img_bytes.getvalue(),
                        "disease_prediction": disease,
                        "confidence": confidence,
                        "upload_date": datetime.datetime.now(datetime.UTC)
                    })
                except Exception as e:
                    st.warning(f"Could not save result: {str(e)}")
            
            # Feedback
            feedback = st.radio(
                "Was this accurate?",
                ["Select option", "👍 Accurate", "👎 Inaccurate"],
                key=f"feedback_{source_type}"
            )
            if feedback != "Select option":
                insert_feedback({
                    "user_id": st.session_state.user_id if st.session_state.logged_in else "anonymous",
                    "prediction": disease,
                    "feedback": feedback,
                    "timestamp": datetime.datetime.now(datetime.UTC)
                })
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# ================
# USER SETTINGS
# ================
# def show_user_settings():
#     """User account management section"""
#     st.header("⚙️ Account Settings")
    
#     with st.expander("🔒 Change Password"):
#         with st.form("change_password"):
#             current = st.text_input("Current Password", type="password")
#             new = st.text_input("New Password", type="password")
#             confirm = st.text_input("Confirm New Password", type="password")
            
#             if st.form_submit_button("Update Password"):
#                 user = find_user_by_username(st.session_state.username)
#                 if user and user["password"] == hash_password(current):
#                     is_strong, msg = is_strong_password(new)
#                     if not is_strong:
#                         st.error(msg)
#                     elif new != confirm:
#                         st.error("Passwords don't match")
#                     else:
#                         update_user_password(st.session_state.user_id, hash_password(new))
#                         st.success("Password updated successfully!")
#                 else:
#                     st.error("Current password is incorrect")

def show_user_settings():
    """User account management section"""
    st.header("⚙️ Account Settings")
    
    # Generate unique key using session state
    form_key = f"change_password_{st.session_state.user_id}"
    
    with st.expander("🔒 Change Password"):
        with st.form(key=form_key):  # Unique key per user
            current = st.text_input("Current Password", type="password")
            new = st.text_input("New Password", type="password")
            confirm = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Password"):
                success, message = update_user_password(
                    st.session_state.user_id,
                    current,
                    new
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    with st.expander("🗑️ Clear Upload History"):
        st.warning("This will permanently delete all your upload history")
        if st.button("Clear All Uploads", key="clear_uploads"):
            clear_user_uploads(st.session_state.user_id)
            st.success("Upload history cleared!")

    
    # Danger zone with custom styling
    st.markdown("""
    <style>
    .danger-zone {
        border: 1px solid #ff4d4d;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
    st.subheader("🚨 Danger Zone")
    st.warning("These actions are irreversible")
    
    with st.form("delete_account"):
        password = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("❌ Delete My Account", type="primary"):
            user = find_user_by_username(st.session_state.username)
            if user and user["password"] == hash_password(password):
                delete_user_account(st.session_state.user_id)
                st.session_state.logged_in = False
                st.rerun()
            else:
                st.error("Incorrect password")
    st.markdown('</div>', unsafe_allow_html=True)

# ================
# MAIN APP LAYOUT
# ================
st.title("🍃 CropGuard - Plant Disease Detection")

# Home Page (Before Login)
if not st.session_state.logged_in:
    st.markdown("""
    ## 🌱 Welcome to CropGuard!
    **AI-powered plant disease detection for farmers and gardeners**
    
    Our system can identify common diseases in:
    - Tomatoes 🍅
    - Potatoes 🥔 
    - Corn 🌽
    - Rice 🌾
    
    ⚠️ **Important Limitations:**
    - Currently supports only the specific diseases listed in our database
    - Works best with clear, well-lit photos of leaves
    - Cannot detect all possible plant diseases or nutrient deficiencies
    """)
    
    # Example images
    st.subheader("📸 Example Detections")
    cols = st.columns(3)
    examples = {
        "Healthy Potato": "Assets/PotatoHealthy(2161).JPG",
        "Tomato Septoria Leaf Spot": "Assets/TomatoSeptoriaLeafSpot(3628).JPG",
        "Corn Common Rust": "Assets/CornCommonRust(3279).JPG"
    }
    
    for col, (name, path) in zip(cols, examples.items()):
        with col:
            try:
                st.image(path, caption=name, use_container_width=True)
            except:
                st.info(f"Example image not available: {name}")

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
        
        # Navigation
        st.session_state.current_tab = st.radio(
            "Navigation",
            ["Home", "Detect", "Settings"],
            index=["Home", "Detect", "Settings"].index(st.session_state.current_tab)
        )
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        
        # Recent history
        st.subheader("📚 Your History")
        uploads = get_user_uploads(st.session_state.user_id, limit=3)
        if uploads:
            for upload in uploads:
                with st.expander(f"{upload.get('disease_prediction', 'Unknown')} ({upload.get('confidence', 0):.0%})"):
                    try:
                        st.image(upload['image'], use_container_width=True)
                    except:
                        st.warning("Image unavailable")
        else:
            st.info("No detection history")

    # Main content
    if st.session_state.current_tab == "Home":
        st.header("🌿 Getting Started")
        st.markdown("""
        ### How to Use CropGuard:
        1. **Capture** clear photos of plant leaves
        2. **Upload** images through our detection interface
        3. **Receive** instant diagnosis and treatment advice
        
        ### Supported Plants & Diseases:
        - **Tomatoes**: Leaf Mold, Yellow Curl Virus, Septoria Spot
        - **Potatoes**: Late Blight, Early Blight, Scab
        - **Corn**: Northern Leaf Blight, Common Rust, Gray Spot
        - **Rice**: Blast, Bacterial Blight, Brown Spot
        
        ⚠️ **Note**: This tool cannot diagnose all plant health issues. 
        For unknown conditions, consult a plant pathologist.
        """)
        
    elif st.session_state.current_tab == "Detect":
        tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera Capture"])
        
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
    
    elif st.session_state.current_tab == "Settings":
        show_user_settings()