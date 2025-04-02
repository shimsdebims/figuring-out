import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

# Load environment variables
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client.get_database()  # Gets database from connection string

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
        result = db.uploads.insert_one(upload_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)

def get_user_uploads(user_id):
    return list(db.uploads.find({"user_id": user_id}).sort("upload_date", -1))