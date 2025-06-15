import React, { useState } from 'react';
import './Classifier.css';
import '../../auth/aws-exports'

const apiUrl = import.meta.env.VITE_API_URL;

function Classifier() {
  const [fileName, setFileName] = useState('Click to upload file');
  const [fileType, setFileType] = useState('nii'); // 'nii' or 'image'
  const [prediction, setPrediction] = useState('None yet');
  const [imageData, setImageData] = useState(null);

  const classifyMRI = async (file, orientation) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('orientation', orientation);

    const response = await fetch(`${apiUrl}/classify`, {
      method: 'POST',
      body: formData,
      headers: {
        "ngrok-skip-browser-warning": 1,
      },
    });

    if (!response.ok) throw new Error('Failed to classify MRI');
    return await response.json();
  };

  const classifyImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${apiUrl}/classify/image`, {
      method: 'POST',
      body: formData,
      headers: {
        "ngrok-skip-browser-warning": 1,
      },
    });

    if (!response.ok) throw new Error('Failed to classify image');
    return await response.json();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const isNii = fileType === 'nii';
    const validNii = file.name.endsWith('.nii.gz');
    const validImg = file.type.startsWith('image/');

    if ((isNii && !validNii) || (!isNii && !validImg)) {
      setFileName('Invalid file');
      setPrediction('Error: Invalid file type');
      setImageData(null);
      return;
    }

    setFileName(file.name);
    setPrediction('Classifying...');
    setImageData(null);

    try {
      let result;
      if (fileType === 'nii') {
        result = await classifyMRI(file, 'Axial');
      } else {
        result = await classifyImage(file);
      }

      setPrediction(result.label);
      if (result.image) setImageData(result.image);
    } catch (err) {
      console.error(err);
      setPrediction('Error: Could not get prediction');
    }
  };

  const handleFileTypeChange = (e) => {
    setFileType(e.target.value);
    setFileName('Click to upload file');
    setPrediction('None yet');
    setImageData(null);
  };

  return (
    <div className="classifier-container">
      <div className="classifier-box">

        {/* Left column */}
        <div className="left-column">

          <div className="dropdown">
            <label htmlFor="file-type-select">File Type:</label>
            <select id="file-type-select" value={fileType} onChange={handleFileTypeChange}>
              <option value="nii">MRI Scan (.nii.gz)</option>
              <option value="image">Image (.png, .jpg)</option>
            </select>
          </div>

          <label htmlFor="file-upload" className="upload-area">
            <span>{fileName}</span>
            <input
              id="file-upload"
              type="file"
              accept={fileType === 'nii' ? '.nii.gz' : 'image/png, image/jpeg'}
              onChange={handleFileChange}
            />
          </label>
        </div>

        {/* Right column */}
        <div className="right-column">
          <p className="prediction-label">
            Prediction: <strong>{prediction}</strong>
          </p>
          {imageData && (
            <img
              src={`data:image/png;base64,${imageData}`}
              alt="Result"
              className="mri-image"
            />
          )}
        </div>

      </div>
    </div>
  );
}

export default Classifier;
