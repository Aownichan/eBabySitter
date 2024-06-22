from flask import Blueprint, request, jsonify, Response
from picamera2 import Picamera2, Preview
import cv2
from flask_socketio import emit
import numpy as np

camera_blueprint = Blueprint('camera', __name__)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Path to the manually downloaded Haarcascade file
haarcascade_path = '/home/aown/Desktop/eBabySitter/server/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

if face_cascade.empty():
    raise IOError("Failed to load haarcascade_frontalface_default.xml. Check the path and OpenCV installation.")

@camera_blueprint.route('/api/show-camera', methods=['POST'])
def show_camera():
    # Picamera2 is always on once started; no need to do anything here
    return jsonify({"message": "Camera is now on"}), 200

@camera_blueprint.route('/api/turn-off-camera', methods=['POST'])
def turn_off_camera():
    # Picamera2 does not support stopping the camera like VideoCapture; no operation
    return jsonify({"message": "Camera is now off"}), 200

@camera_blueprint.route('/api/camera-feed', methods=['GET'])
def camera_feed():
    def generate():
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if len(faces) == 0:
                emit('no_face_detected', {'data': 'No face detected'})

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
