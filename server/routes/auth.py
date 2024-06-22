from flask import Blueprint, request, jsonify
from flask_bcrypt import Bcrypt
from models.user import User
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt() 


auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # Check if user already exists
    if User.get(email):
        return jsonify({"message": "User already exists"}), 400

    # Hash password and insert into database
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    User.insert(email, hashed_password)
    return jsonify({"message": "User registered successfully"}), 201
@auth_blueprint.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # Check if user exists and password is correct
    user = User.get(email)
    if user and bcrypt.check_password_hash(user[1], password):
        return jsonify({"message": "Logged in successfully", "email": email}), 200
    else:
        return jsonify({"message": "Invalid email or password"}), 401
