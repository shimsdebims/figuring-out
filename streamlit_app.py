import streamlit as st
import datetime
from PIL import Image
from auth import register_user, login_user
from database import insert_upload, get_user_uploads
import os

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Authentication sidebar - SIMPLIFIED AND GUARANTEED TO SHOW BOTH OPTIONS
with st.sidebar:
    st.title("User Account")
    
    if not st.session_state.logged_in:
        # Create two columns instead of tabs for absolute visibility
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
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
                    st.warning("Need both fields")
        
        with col2:
            st.subheader("Register")
            reg_username = st.text_input("New username", key="reg_username")
            reg_password = st.text_input("New password", type="password", key="reg_password")
            reg_email = st.text_input("Email", key="reg_email")
            
            if st.button("Register"):
                if reg_username and reg_password and reg_email:
                    success, message = register_user(reg_username, reg_password, reg_email)
                    if success:
                        st.success("Registered! Please login.")
                    else:
                        st.error(message)
                else:
                    st.warning("All fields required")
    
    else:
        st.write(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.success("Logged out!")
            st.rerun()
        
        st.subheader("Your Uploads")
        uploads = get_user_uploads(st.session_state.user_id)
        if uploads:
            for upload in uploads:
                st.write(f"{upload['disease_prediction']} ({upload['upload_date'].strftime('%Y-%m-%d')})")
        else:
            st.info("No uploads yet")

# Main app
st.title("Crop Disease Detection")

if st.session_state.logged_in:
    uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            if st.button("Analyze"):
                # Replace with your actual prediction code
                disease = "Tomato - Healthy"  # Example
                confidence = 0.95  # Example
                
                # Save to database
                upload_data = {
                    "user_id": st.session_state.user_id,
                    "image_path": f"uploads/{st.session_state.username}/{uploaded_file.name}",
                    "disease_prediction": disease,
                    "confidence": confidence,
                    "upload_date": datetime.datetime.utcnow()
                }
                insert_upload(upload_data)
                
                st.success(f"Detected: {disease}")
                st.write(f"Confidence: {confidence:.0%}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Please login or register above")