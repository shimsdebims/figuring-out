import hashlib
import datetime
from database import find_user_by_username, insert_user

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    if find_user_by_username(username):
        return False, "Username already exists"
    hashed_password = hash_password(password)
    user_data = {
        "username": username,
        "password": hashed_password,
        "email": email,
        "created_at": datetime.datetime.utcnow()
    }
    return insert_user(user_data)

def login_user(username, password):
    user = find_user_by_username(username)
    if not user or user["password"] != hash_password(password):
        return False, "Invalid username or password"
    return True, str(user["_id"])