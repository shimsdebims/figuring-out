import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

# Load environment variables
load_dotenv()

# Initialize connection
def get_db():
    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client.get_database()  # Gets database specified in the URI
        return db
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

# Simplified database functions
def insert_user(user_data):
    try:
        db = get_db()
        result = db.users.insert_one(user_data)
        return True, str(result.inserted_id)
    except DuplicateKeyError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)

def find_user_by_username(username):
    try:
        db = get_db()
        return db.users.find_one({"username": username})
    except Exception as e:
        print(f"Error finding user: {e}")
        return None

def insert_upload(upload_data):
    try:
        db = get_db()
        result = db.uploads.insert_one(upload_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)

def get_user_uploads(user_id):
    try:
        db = get_db()
        return list(db.uploads.find({"user_id": user_id}).sort("upload_date", -1))
    except Exception as e:
        print(f"Error getting uploads: {e}")
        return []