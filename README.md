# Streamlit-webapp
Plant disease detection Web app
A Streamlit web application for detecting plant diseases with user authentication.


Users upload images of plants and the app can detect what disease, how to fix it, and a fun fact is displayed

The crops this app includes are:
Tomato
Potato
Rice
Maize


## Features
- User registration and login system
- Image upload for disease detection
- Detailed disease information including:
  - Symptoms
  - Treatment recommendations
  - Fun facts
- Data persistence with MongoDB


## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Create a `.env` file with your MongoDB URI
3. Run the app: `streamlit run streamlit_app.py`

## Database Structure
- `users` collection: Stores user credentials (username and hashed passwords)
- `uploads` collection: Stores user uploads with predictions
[file content end]