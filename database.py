import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError
from bson.binary import Binary
from PIL import Image
import io
import base64
import json
import datetime
import time
import random
import string
import hashlib

# Load environment variables
load_dotenv()

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["cropDiseaseDB"]

def initialize_db():
    """Create necessary indexes"""
    db.users.create_index([("username", pymongo.ASCENDING)], unique=True)
    db.uploads.create_index([("user_id", pymongo.ASCENDING)])
    db.uploads.create_index([("upload_date", pymongo.DESCENDING)])
    ensure_upload_dirs()


def insert_user(user_data):
    try:
        result = db.users.insert_one(user_data)
        return True, str(result.inserted_id)
    except DuplicateKeyError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)

def find_user_by_username(username):
    return db.users.find_one({"username": username})

def insert_upload(upload_data):
    try:
        # Convert image to binary
        if 'image' in upload_data:
            upload_data['image_binary'] = Binary(upload_data['image'])
            del upload_data['image']
        
        result = db.uploads.insert_one(upload_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)

def get_user_uploads(user_id, limit=5):
    """Get user uploads with image objects"""
    uploads = list(db.uploads.find({"user_id": user_id})
                  .sort("upload_date", -1)
                  .limit(limit))
    
    # Convert binary to image
    for upload in uploads:
        if 'image_binary' in upload:
            try:
                upload['image'] = Image.open(io.BytesIO(upload['image_binary']))
            except Exception as e:
                upload['image_error'] = str(e)
    
    return uploads

def insert_feedback(feedback_data):
    try:
        result = db.feedback.insert_one(feedback_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)
    
    def ensure_upload_dirs():
    """Ensure upload directories exist"""
    os.makedirs("uploads", exist_ok=True)