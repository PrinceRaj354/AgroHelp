Hero.jsx

import React from 'react';
import '../../css/Hero.css';
import { Link } from 'react-router-dom';



// Import images
import CropImage from '../../assets/CROP.PNG';
import FertilizerImage from '../../assets/FERTILIZER.PNG';
import DiseaseImage from '../../assets/DISEASES.PNG';

function Hero() {
  return (
    <div className="hero">
      {/* Hero Title and Description */}
      <div className="heading">
        <h1 className="hero-title">Empowering Farmers for a Greener Future</h1>
        <p className="hero-description">
          Discover sustainable farming practices, get crop recommendations, and learn how to maintain plant health.
        </p>
      </div>

      {/* Hero Buttons Section */}
      <div className="hero-buttons">
        <Link to="/crop-recommendation">
          <div
            className="hero-button"
            style={{
              backgroundImage: `url(${CropImage})`, // Correct way of using imported images
            }}
          ></div>
        </Link>
        <Link to="/fertilizer-suggestion">
          <div
            className="hero-button"
            style={{
              backgroundImage: `url(${FertilizerImage})`,
            }}
          ></div>
        </Link>
        <Link to="/plant-disease-detection">
          <div
            className="hero-button"
            style={{
              backgroundImage: `url(${DiseaseImage})`,
            }}
          ></div>
        </Link>
      </div>

      {/* New Simple Text Section */}
      <div className="simple-text-section">
        <h7>About AGROHELP</h7>
        <p>
          AgroHelp is a technology-driven platform designed to address critical agricultural challenges like inefficient crop selection, improper fertilizer use, and undetected crop diseases. By leveraging advanced AI and machine learning, AgroHelp provides tailored crop recommendations based on soil and environmental data, precise fertilizer suggestions to optimize costs and improve soil health, and AI-powered disease detection through image recognition for timely intervention. With an intuitive, web-based interface, AgroHelp empowers farmers to enhance productivity, reduce costs, and adopt sustainable farming practices, ensuring long-term agricultural profitability and food security.
        </p>
      </div>
    </div>
  );
}

export default Hero;




