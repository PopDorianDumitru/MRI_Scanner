import React, { useState } from 'react';
import './Generator.css';

const apiUrl = import.meta.env.VITE_API_URL;


function Generator() {
  const [count, setCount] = useState(1);
  const [type, setType] = useState('Normal');
  const [images, setImages] = useState([]);

  const handleGenerate = async () => {
    const formData = new FormData();
    formData.append('label', type);
    formData.append('count', count);
    try {
        const res = await fetch(`${apiUrl}/generate`, {
        method: 'POST',
        body: formData,
        headers: {
          "ngrok-skip-browser-warning": 1
        }
        });

        if (!res.ok) {
            throw new Error('Failed to generate images');
        }

        const data = await res.json();

        // Map base64 strings to image objects
        const newImages = data.images.map((b64, i) => ({
            id: i,
            src: `data:image/png;base64,${b64}`,
        }));

        setImages(newImages);
    } catch (err) {
        console.error(err);
        alert('Error: ' + err.message);
    }
};


  return (
    <div className="generator-container">
      <div className="generator-box">

        {/* Left Column */}
        <div className="left-column">
          <div className="form-group">
            <label htmlFor="image-count">Number of Images: {count}</label>
            <input
              id="image-count"
              type="range"
              min="0"
              max="5"
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label htmlFor="dementia-type">Dementia Type:</label>
            <select
              id="dementia-type"
              value={type}
              onChange={(e) => setType(e.target.value)}
            >
              <option value="Normal">Normal</option>
              <option value="Mild Dementia">Mild Dementia</option>
              <option value="Moderate Dementia">Moderate Dementia</option>
              <option value="Severe Dementia">Severe Dementia</option>
              <option value="Very Severe Dementia">Very Severe Dementia</option>
            </select>
          </div>

          <button onClick={handleGenerate}>Generate</button>
        </div>

        {/* Right Column */}
        <div className="right-column">
          <div className="image-grid">
            {images.map((img) => (
              <img
                key={img.id}
                src={img.src}
                alt={`Generated ${type} ${img.id + 1}`}
                width="128"
                height="128"
              />
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}

export default Generator;
