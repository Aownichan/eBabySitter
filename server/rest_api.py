from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import random
import cv2
import pygame
from werkzeug.utils import secure_filename
from models.user import User
from routes.auth import auth_blueprint
from routes.camera import camera_blueprint
from routes.song import song_blueprint 
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt() 

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
bcrypt = Bcrypt(app)

app.register_blueprint(auth_blueprint)
app.register_blueprint(camera_blueprint)
app.register_blueprint(song_blueprint)

if __name__ == '__main__': 
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)