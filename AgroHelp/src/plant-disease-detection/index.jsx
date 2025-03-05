/*import React, { useState } from "react";
import axios from "axios";
import './PlantDiseaseDetector.css'; // Import the CSS file

const PlantDiseaseDetector = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction("");
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an image file.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(response.data.predicted_class);
      setError("");
    } catch (err) {
      setError("Failed to predict. Please try again.");
    }
  };

  return (
    <div className="container">
      <h4>Plant Disease Detector</h4>
      <form className="form" onSubmit={handleSubmit}>
        <div>
          <label htmlFor="fileInput" className="label">Upload an image of a plant:</label>
          <input type="file" id="fileInput" accept="image/*" onChange={handleFileChange} />
        </div>
        <div className="form-container">
          <button type="submit">Predict</button>
        </div>
      </form>
      {prediction && (
        <div className="message prediction">
          <p1>Prediction: {prediction}</p1>
        </div>
      )}
      {error && (
        <div className="message error">
          <p1>Error: {error}</p1>
        </div>
      )}
    </div>
    
  );
};

export default PlantDiseaseDetector;
*/


import React, { useState } from "react";
import axios from "axios";
import './PlantDiseaseDetector.css'; // Import the CSS file

const PlantDiseaseDetector = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreview(null); // Clear preview until prediction is made
    setPrediction("");
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an image file.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(response.data.predicted_class);
      setPreview(URL.createObjectURL(selectedFile)); // Generate the preview only after successful prediction
      setError("");
    } catch (err) {
      setError("Failed to predict. Please try again.");
    }
  };

  return (
    <div className="container">
      <h4>Plant Disease Detector</h4>
      <form className="form" onSubmit={handleSubmit}>
        <div>
          <label htmlFor="fileInput" className="label">Upload an image of a plant:</label>
          <input type="file" id="fileInput" accept="image/*" onChange={handleFileChange} />
        </div>
        <div className="form-container">
          <button type="submit">Predict</button>
        </div>
      </form>
      {prediction && (
        <div className="message prediction">
          <p>Prediction: {prediction}</p>
        </div>
      )}
      {error && (
        <div className="message error">
          <p>Error: {error}</p>
        </div>
      )}
      {preview && prediction && (
        <div className="image-preview">
          <img src={preview} alt="Uploaded prediction" />
        </div>
      )}
    </div>
  );
};

export default PlantDiseaseDetector;
