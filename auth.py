import hashlib
import re
import datetime
from database import find_user_by_username, insert_user

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email):
    """Basic email format validation"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def is_strong_password(password):
    """Check password meets requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain 1 uppercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain 1 number" 
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain 1 special character"
    return True, ""

def register_user(username, password, email):
    """Register new user with validation"""
    # Check password strength
    is_strong, msg = is_strong_password(password)
    if not is_strong:
        return False, msg
        
    if find_user_by_username(username):
        return False, "Username already exists"
        
    hashed_password = hash_password(password)
    user_data = {
        "username": username,
        "password": hashed_password,
        "email": email,
        "created_at": datetime.datetime.now(datetime.UTC)
    }
    return insert_user(user_data)

def login_user(username, password):
    """Authenticate user"""
    user = find_user_by_username(username)
    if not user or user["password"] != hash_password(password):
        return False, "Invalid username or password"
    return True, str(user["_id"])

def update_user_password(user_id, current_password, new_password):
    """Secure password update with validation"""
    from database import find_user_by_id, db
    from bson.objectid import ObjectId
    
    # 1. Verify current password
    user = find_user_by_id(user_id)
    if not user or user["password"] != hash_password(current_password):
        return False, "Current password is incorrect"
    
    # 2. Validate new password
    is_strong, msg = is_strong_password(new_password)
    if not is_strong:
        return False, msg
    
    # 3. Update password
    try:
        db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"password": hash_password(new_password)}}
        )
        return True, "Password updated successfully"
    except Exception as e:
        return False, str(e)