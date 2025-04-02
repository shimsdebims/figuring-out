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
if 'auth_option' not in st.session_state:
    st.session_state.auth_option = "Login"

# Main title
st.title("Crop Disease Detection")

# Authentication sidebar
with st.sidebar:
    st.title("User Account")
    
    if not st.session_state.logged_in:
        # Auth option selection
        auth_option = st.radio("", ("Login", "Register"), horizontal=True, key="auth_radio")
        
        if auth_option == "Login":
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
        
        else:  # Register option
            st.subheader("Register")
            reg_username = st.text_input("New username", key="reg_username")
            reg_password = st.text_input("New password", type="password", key="reg_password")
            reg_email = st.text_input("Email", key="reg_email")
            
            if st.button("Register", key="register_button"):
                if reg_username and reg_password and reg_email:
                    success, message = register_user(reg_username, reg_password, reg_email)
                    if success:
                        st.success("Registered! Please login.")
                        # Switch to login tab after successful registration
                        st.session_state.auth_option = "Login"
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
            st.success("Logged out!")
            st.rerun()
        
        st.subheader("Your Uploads")
        uploads = get_user_uploads(st.session_state.user_id)
        if uploads:
            for upload in uploads:
                st.write(f"{upload['disease_prediction']} ({upload['upload_date'].strftime('%Y-%m-%d')})")
        else:
            st.info("No uploads yet")

# Main app content
if st.session_state.logged_in:
    uploaded_file = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            if st.button("Analyze"):
                # Ensure upload directory exists
                os.makedirs(f"uploads/{st.session_state.username}", exist_ok=True)
                
                # Save the image file
                image_path = f"uploads/{st.session_state.username}/{uploaded_file.name}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Replace with your actual prediction code
                disease = "Tomato - Healthy"  # Example
                confidence = 0.95  # Example
                
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
                import json
                try:
                    with open("disease_info.json", "r") as f:
                        disease_info = json.load(f)
                        
                    if disease in disease_info:
                        info = disease_info[disease]
                        st.subheader("Disease Information")
                        st.write(f"**Symptoms:** {info['symptoms']}")
                        st.write(f"**Treatment:** {info['treatment']}")
                        st.write(f"**Fun Fact:** {info['fun_fact']}")
                except Exception as e:
                    st.warning(f"Could not load disease information: {str(e)}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Please login or register in the sidebar to detect crop diseases")