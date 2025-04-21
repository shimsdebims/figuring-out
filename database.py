import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError
from bson.binary import Binary
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Database connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["cropDiseaseDB"]

def ensure_upload_dirs():
    """Ensure upload directories exist"""
    os.makedirs("uploads", exist_ok=True)

def initialize_db():
    """Initialize database indexes"""
    db.users.create_index([("username", pymongo.ASCENDING)], unique=True)
    db.uploads.create_index([("user_id", pymongo.ASCENDING)])
    db.uploads.create_index([("upload_date", pymongo.DESCENDING)])
    ensure_upload_dirs()

def get_image_bytes(upload_data):
    """Safely extract image bytes from upload data"""
    if 'image_binary' in upload_data:
        return upload_data['image_binary']
    return upload_data.get('image', b'')

def get_user_uploads(user_id, limit=5):
    """Get user uploads with image processing"""
    uploads = list(db.uploads.find({"user_id": user_id})
                  .sort("upload_date", -1)
                  .limit(limit))
    
    for upload in uploads:
        img_bytes = get_image_bytes(upload)
        if img_bytes:
            try:
                upload['image'] = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                upload['image_error'] = str(e)
    return uploads

def insert_feedback(feedback_data):
    try:
        result = db.feedback.insert_one(feedback_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)