import hashlib
import datetime
from database import find_user_by_username, insert_user
import re
from password_strength import PasswordPolicy
from password_strength import PasswordStats
import requests


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password, email):
    # Check password strength first
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

# Password policy
policy = PasswordPolicy.from_names(
    length=8,  # min length
    uppercase=1,  # need min. 1 uppercase letters
    numbers=1,  # need min. 1 digits
    special=1,  # need min. 1 special characters
)

def is_strong_password(password):
    """Check if password meets requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least 1 uppercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least 1 number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least 1 special character"
    return True, ""

def is_valid_email(email):
    """Basic email format validation"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def is_disposable_email(email):
    """Check against known disposable email providers"""
    domain = email.split('@')[-1]
    try:
        response = requests.get(f"https://disposable.debounce.io/?email={domain}")
        return response.json().get('disposable', False)
    except:
        return False  # If API fails, assume valid

# Update register_user function
def register_user(username, password, email):
    # Email validation
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if is_disposable_email(email):
        return False, "Disposable emails not allowed"

def login_user(username, password):
    user = find_user_by_username(username)
    if not user or user["password"] != hash_password(password):
        return False, "Invalid username or password"
    return True, str(user["_id"])
