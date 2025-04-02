import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure, DuplicateKeyError, OperationFailure

# Load environment variables from .env file
load_dotenv()

def get_database():
    """Connect to MongoDB and return the database"""
    try:
        # Get URI from environment variables
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise ValueError("MongoDB URI not found in environment variables")
        
        # Connect to MongoDB with timeout settings
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        
        # Verify the connection works
        client.admin.command('ping')
        return client.get_database()  # This will use the database specified in the URI
        
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

def init_db():
    """Initialize database indexes (skip collection creation since they exist)"""
    try:
        db = get_database()
        
        # Check if collections exist and create indexes
        if "users" in db.list_collection_names():
            db.users.create_index([("username", pymongo.ASCENDING)], unique=True)
        
        if "uploads" in db.list_collection_names():
            db.uploads.create_index([("user_id", pymongo.ASCENDING)])
        
        print("Database indexes initialized successfully")
        return True
    
    except OperationFailure as e:
        print(f"Database operation failed (you may need admin privileges): {e}")
        return False
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def find_user_by_username(username):
    """Find a user by username"""
    try:
        db = get_database()
        return db.users.find_one({"username": username})
    except Exception as e:
        print(f"Error finding user: {e}")
        return None

def insert_user(user_data):
    """Insert a new user into the database"""
    try:
        db = get_database()
        result = db.users.insert_one(user_data)
        return True, str(result.inserted_id)
    except DuplicateKeyError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)

def insert_upload(upload_data):
    """Insert an upload record into the database"""
    try:
        db = get_database()
        result = db.uploads.insert_one(upload_data)
        return True, str(result.inserted_id)
    except Exception as e:
        return False, str(e)

def get_user_uploads(user_id):
    """Get all uploads for a specific user"""
    try:
        db = get_database()
        uploads = list(db.uploads.find(
            {"user_id": user_id},
            {"_id": 0}  # Exclude MongoDB _id field from results
        ).sort("upload_date", pymongo.DESCENDING))
        return uploads
    except Exception as e:
        print(f"Error getting user uploads: {e}")
        return []