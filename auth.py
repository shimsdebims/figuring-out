import hashlib
import re
import datetime
from database import find_user_by_username, insert_user

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def find_user_by_username(username):
    """Find user by username"""
    return db.users.find_one({"username": username})

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
    try:
        # Input validation
        if not username or not password or not email:
            return False, "All fields are required"
            
        # Check password strength
        is_strong, msg = is_strong_password(password)
        if not is_strong:
            return False, msg
            
        # Check if user exists
        if find_user_by_username(username):
            return False, "Username already exists"
            
        # Validate email
        if not is_valid_email(email):
            return False, "Invalid email format"
            
        # Create user
        hashed_password = hash_password(password)
        user_data = {
            "username": username,
            "password": hashed_password,
            "email": email,
            "created_at": datetime.datetime.now(datetime.UTC)
        }
        
        success, user_id = insert_user(user_data)
        if not success:
            return False, user_id  # user_id contains error message here
            
        return True, "Registration successful"
        
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def login_user(username, password):
    """Authenticate user"""
    try:
        if not username or not password:
            return False, "Username and password required"
            
        user = find_user_by_username(username)
        if not user:
            return False, "Invalid username or password"
            
        if user["password"] != hash_password(password):
            return False, "Invalid username or password"
            
        return True, str(user["_id"])
        
    except Exception as e:
        return False, f"Login error: {str(e)}"