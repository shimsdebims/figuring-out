import hashlib
import datetime
from database import find_user_by_username, insert_user

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register a new user."""
    # Check if username already exists
    existing_user = find_user_by_username(username)
    if existing_user:
        return False, "Username already exists"
    
    # Create new user
    hashed_password = hash_password(password)
    user_data = {
        "username": username,
        "password": hashed_password,
        "email": email,
        "created_at": datetime.datetime.utcnow()
    }
    
    success, message = insert_user(user_data)
    return success, message

def login_user(username, password):
    """Verify username and password."""
    user = find_user_by_username(username)
    
    if not user:
        return False, "Invalid username or password"
    
    hashed_password = hash_password(password)
    if user["password"] != hashed_password:
        return False, "Invalid username or password"
    
    return True, str(user["_id"])