import React, { useState } from 'react';

const AudioFileInput = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileError, setFileError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) {
      setFileError('No file selected');
      return;
    }
    if (file.type !== 'audio/mpeg' && file.type !== 'audio/wav') {
      setFileError('Only MP3 and WAV files are supported');
      return;
    }
    setSelectedFile(file);
    setFileError(null);
  };

  const handleUpload = (file) => {
    const formData = new FormData();
    formData.append('file', file);
  
    fetch('http://localhost:5000/save', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          console.error(data.error);
        } else {
          console.log(`File uploaded successfully: ${data.filename}`);
        }
      })
      .catch((error) => {
        console.error('Error uploading file:', error);
      });
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      {selectedFile && (
        <div>
          <p>Selected file: {selectedFile.name}</p>
          <button onClick={() => handleUpload(selectedFile)}>Upload file</button>
        </div>
      )}
      {fileError && <p style={{ color: 'red' }}>{fileError}</p>}
    </div>
  );
};

export default AudioFileInput;