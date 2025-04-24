import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError
from bson.binary import Binary
from PIL import Image
import io
from bson.objectid import ObjectId 

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

def find_user_by_username(username):
    """Find user by username"""
    return db.users.find_one({"username": username})


def insert_user(user_data):
    """Insert new user into database"""
    try:
        result = db.users.insert_one(user_data)
        return True, str(result.inserted_id)
    except DuplicateKeyError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)

def insert_upload(upload_data):
    """Insert upload data into database"""
    try:
        result = db.uploads.insert_one(upload_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)

# User management functions
def update_user_password(user_id, new_hashed_pwd):
    """Update user password"""
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password": new_hashed_pwd}}
    )

def clear_user_uploads(user_id):
    """Delete all uploads for a user"""
    db.uploads.delete_many({"user_id": user_id})

def delete_user_account(user_id):
    """Completely remove user account and data"""
    with client.start_session() as session:
        with session.start_transaction():
            db.users.delete_one({"_id": ObjectId(user_id)}, session=session)
            db.uploads.delete_many({"user_id": user_id}, session=session)

        
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
