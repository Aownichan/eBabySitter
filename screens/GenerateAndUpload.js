import React, { useState, useRef } from 'react';
import axios from 'axios';

const RecordAndUpload = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [message, setMessage] = useState('');
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const startRecording = async () => {
        setMessage('');
        setIsRecording(true);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };
            mediaRecorder.start();
        } catch (err) {
            console.error(err);
            setMessage('Error accessing microphone.');
        }
    };

    const stopRecording = () => {
        setIsRecording(false);
        const mediaRecorder = mediaRecorderRef.current;
        if (mediaRecorder) {
            mediaRecorder.stop();
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                audioChunksRef.current = [];
                await uploadAudio(audioBlob);
            };
        }
    };

    const uploadAudio = async (audioBlob) => {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recorded_audio.wav');

        try {
            const response = await axios.post('http://localhost:5000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setMessage(response.data.message);
        } catch (error) {
            console.error(error);
            setMessage('File upload failed.');
        }
    };

    return (
        <div>
            <button onClick={isRecording ? stopRecording : startRecording}>
                {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>
            {message && <p>{message}</p>}
        </div>
    );
};

export default RecordAndUpload;
