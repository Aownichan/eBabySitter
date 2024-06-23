import os
from pydub import AudioSegment
from pydub.playback import play

def play_audio(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Play the audio file
    play(audio)

# Ensure the path is correct
file_path = 'C:/Users/Floppa Worshipper/MyProject/uploads/recorded_audio.wav'
play_audio(file_path)
