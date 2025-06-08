import React, { useState } from 'react';
import './Classifier.css';

const apiUrl = import.meta.env.VITE_API_URL;

function Classifier() {
  const [fileName, setFileName] = useState('Click to upload .nii.gz');
  const [pov, _] = useState('Axial');
  const [prediction, setPrediction] = useState('None yet');
  const [imageData, setImageData] = useState(null);

  const classifyMRI = async (file, orientation) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('orientation', orientation);

    const response = await fetch(`${apiUrl}/classify`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Failed to classify MRI');
    }

    const data = await response.json();
    return {
      label: data.label,
      image: data.image,
    };
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file || !file.name.endsWith('.nii.gz')) {
        setFileName('Invalid file');
        setPrediction('Error: Please upload a valid .nii.gz file');
        setImageData(null);

        return;
    }

    setFileName(file.name);
    setPrediction('Classifying...');
    setImageData(null);

    try {
        const { label, image } = await classifyMRI(file, pov);
        setPrediction(label);
        setImageData(image);
    } catch (err) {
        console.error(err);
        setPrediction('Error: Could not get prediction');
    }
  };

//   const handlePovChange = (e) => {
//     setPov(e.target.value);
//   };

  return (
    <div className="classifier-container">
      <div className="classifier-box">
        
        {/* Left column */}
        <div className="left-column">
          <label htmlFor="file-upload" className="upload-area">
            <span>{fileName}</span>
            <input
              id="file-upload"
              type="file"
              accept=".nii.gz"
              onChange={handleFileChange}
            />
          </label>

          {/* <div className="dropdown">
            <label htmlFor="view-select">Point of View:</label>
            <select id="view-select" value={pov} onChange={handlePovChange}>
              <option value="Sagittal">Sagittal</option>
              <option value="Coronal">Coronal</option>
              <option value="Axial">Axial</option>
            </select>
          </div> */}
        </div>

        {/* Right column */}
        <div className="right-column">
          <p className="prediction-label">
            Prediction: <strong>{prediction}</strong>
          </p>
          {imageData && (
            <img
              src={`data:image/png;base64,${imageData}`}
              alt="MRI Slice"
              className="mri-image"
            />
          )}
        </div>

      </div>
    </div>
  );
}

export default Classifier;
