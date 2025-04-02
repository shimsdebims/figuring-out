import pymongo
from pymongo import MongoClient
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection string - store this in .env file for security
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://username:password@cluster.mongodb.net/cropDiseaseDB")

def get_database():
    client = MongoClient(MONGO_URI)
    return client["cropDiseaseDB"]  # database name

def init_db():
    # Check if collections exist, if not create them
    db = get_database()
    if "users" not in db.list_collection_names():
        # Create users collection with username index
        users = db["users"]
        users.create_index([("username", pymongo.ASCENDING)], unique=True)
    
    if "uploads" not in db.list_collection_names():
        # Create uploads collection with user_id index
        uploads = db["uploads"]
        uploads.create_index([("user_id", pymongo.ASCENDING)])

def find_user_by_username(username):
    db = get_database()
    return db.users.find_one({"username": username})

def insert_user(user_data):
    db = get_database()
    try:
        result = db.users.insert_one(user_data)
        return True, str(result.inserted_id)
    except pymongo.errors.DuplicateKeyError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)

def insert_upload(upload_data):
    db = get_database()
    try:
        result = db.uploads.insert_one(upload_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)

def get_user_uploads(user_id):
    db = get_database()
    uploads = list(db.uploads.find(
        {"user_id": user_id}, 
        {"_id": 0}  # Exclude _id from results
    ).sort("upload_date", pymongo.DESCENDING))
    return uploads