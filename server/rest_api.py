from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from picamera2 import Picamera2, Preview
import pygame
import os
import random
import string
from werkzeug.utils import secure_filename 
from routes.auth import auth_blueprint
import cv2

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for WebSocket

show_camera = False

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Load pre-trained face detection model
haarcascade_path = '/home/aown/Desktop/eBabySitter/server/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

app.register_blueprint(auth_blueprint)

def generate_camera_frames():
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration()
    picam2.configure(preview_config)
    picam2.start()
    
    while True:
        if show_camera:
            # Capture the frame
            frame = picam2.capture_array()
            
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw bounding boxes around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Encode frame to JPEG format for streaming
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Send a message to the client if no faces are detected
            if len(faces) == 0:
                socketio.emit('no_face_detected', {'message': 'No face detected'})
        else:
            # Send a placeholder image when camera is off
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('placeholder.jpg', 'rb').read() + b'\r\n')

@app.route('/api/data')
def get_data():
    data = {'message': 'Hello from Python server!'}
    return jsonify(data)

@app.route('/api/show-camera', methods=['POST'])
def toggle_camera():
    global show_camera
    show_camera = True
    return jsonify({'success': True})

@app.route('/api/turn-off-camera', methods=['POST'])
def turn_off_camera():
    global show_camera
    show_camera = False
    return jsonify({'success': True})

@app.route('/api/camera-feed')
def camera_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/play-song')
def play_song():
    sounds_dir = os.path.join(os.path.dirname(__file__), 'sounds')
    songs = [os.path.join(sounds_dir, song) for song in os.listdir(sounds_dir) if song.endswith('.mp3')]

    if not songs:
        return jsonify({'success': False, 'message': 'No songs found in sounds directory'})

    song_to_play = random.choice(songs)
    pygame.mixer.music.load(song_to_play)
    pygame.mixer.music.play()

    return jsonify({'success': True, 'song': os.path.basename(song_to_play)})

@app.route('/api/stop-song', methods=['POST'])
def stop_song():
    pygame.mixer.music.stop()
    return jsonify({'success': True}) 

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  
        print(filepath)  
    
        #play_audio(filepath)
        return jsonify({"message": "File uploaded successfully", "filename": filename}), 200

def generate_random_string_with_extension(length):
    letters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(letters, k=length))
    return random_string + ".mp3"

# save folder configuration
SAVE_FOLDER = 'save'
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
app.config['SAVE_FOLDER'] = SAVE_FOLDER
@app.route('/save', methods=['POST'])
def save_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = generate_random_string_with_extension(6)
        filepath = os.path.join(app.config['SAVE_FOLDER'], filename)
        file.save(filepath)  
        print(filepath)  
    
        #play_audio(filepath)
        return jsonify({"message": "audio saved successfully", "filename": filename}), 200


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
