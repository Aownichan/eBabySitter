from flask import Blueprint, request, jsonify
import os
import pygame
from werkzeug.utils import secure_filename 
import random

song_blueprint = Blueprint('song', __name__)

#pygame.mixer.init()

@song_blueprint.route('/api/play-song', methods=['POST'])
def play_song():
    sounds_folder = 'sounds'
    if not os.path.exists(sounds_folder):
        return jsonify({"message": "No sounds directory found"}), 404

    songs = [f for f in os.listdir(sounds_folder) if f.endswith('.mp3')]
    if not songs:
        return jsonify({"message": "No songs found"}), 404

    song = random.choice(songs)
    pygame.mixer.music.load(os.path.join(sounds_folder, song))
    pygame.mixer.music.play()
    return jsonify({"message": f"Playing {song}"}), 200

@song_blueprint.route('/api/stop-song', methods=['POST'])
def stop_song():
    pygame.mixer.music.stop()
    return jsonify({"message": "Song stopped"}), 200