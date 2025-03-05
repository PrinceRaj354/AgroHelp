import React, { useState } from 'react';
import './crop.css';
import { chatSession } from '../service/AIModal.jsx';

function CropRecommendation() {
  const [soilTexture, setSoilTexture] = useState('');
  const [phLevel, setPhLevel] = useState('');
  const [nitrogen, setNitrogen] = useState('');
  const [phosphorus, setPhosphorus] = useState('');
  const [potassium, setPotassium] = useState('');
  const [temperature, setTemperature] = useState('');
  const [rainfall, setRainfall] = useState('');
  const [humidity, setHumidity] = useState('');
  const [windConditions, setWindConditions] = useState('');
  const [waterSource, setWaterSource] = useState('');
  const [irrigationSystem, setIrrigationSystem] = useState(false);
  const [waterUsage, setWaterUsage] = useState('');
  const [farmingTech, setFarmingTech] = useState('');
  const [prevCrops, setPrevCrops] = useState('');
  const [cropPreferences, setCropPreferences] = useState('');
  const [region, setRegion] = useState('');
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const aiPrompt = `Based on the following data provided by the farmer:
    - Soil Texture: ${soilTexture}
    - Soil pH Level: ${phLevel}
    - Nitrogen Level: ${nitrogen} mg/kg
    - Phosphorus Level: ${phosphorus} mg/kg
    - Potassium Level: ${potassium} mg/kg
    - Average Temperature: ${temperature}°C
    - Seasonal Rainfall: ${rainfall} mm
    - Humidity Level: ${humidity}%
    - Wind Conditions: ${windConditions}
    - Water Source: ${waterSource}
    - Irrigation System Available: ${irrigationSystem ? 'Yes' : 'No'}
    - Water Usage Limitations: ${waterUsage}
    - Farming Technology Access: ${farmingTech}
    - Previously Grown Crops: ${prevCrops || 'None specified'}
    - Crop Preferences: ${cropPreferences || 'None specified'}
    - Region/Place: ${region || 'Not specified'}

    Consider sustainability, efficient land use, environmental impact, and crop rotation. Recommend:
    1. Suitable crops for the conditions.
    2. Suitability explanation.
    3. Expected yields.
    4. Tips for optimal growth.
    5. Any environmental or sustainability considerations.`;

    try {
      const result = await chatSession.sendMessage(aiPrompt);
      const data = await result.response.text();
      console.log("JSON Response:", data);
      const parsedData = JSON.parse(data);

      // Safely set recommendations
      setRecommendations(parsedData.recommendations || [{ error: "No recommendations found." }]);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setRecommendations([{ error: "Unable to generate recommendations. Please try again later." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h6>CROP Recommendation Form</h6>
      <form onSubmit={handleSubmit}>
        {/* Soil and Environmental Inputs */}
        <label>Soil Texture:</label>
        <select value={soilTexture} onChange={(e) => setSoilTexture(e.target.value)} >
          <option value="">Select...</option>
          <option value="Clay">Clay</option>
          <option value="Loam">Loam</option>
          <option value="Sand">Sand</option>
        </select>

        <label>Soil pH Level:</label>
        <select value={phLevel} onChange={(e) => setPhLevel(e.target.value)} >
          <option value="">Select...</option>
          <option value="Acidic">Acidic</option>
          <option value="Neutral">Neutral</option>
          <option value="Alkaline">Alkaline</option>
        </select>

        <label>Nitrogen Level (mg/kg):</label>
        <input type="number" value={nitrogen} onChange={(e) => setNitrogen(e.target.value)} />

        <label>Phosphorus Level (mg/kg):</label>
        <input type="number" value={phosphorus} onChange={(e) => setPhosphorus(e.target.value)} />

        <label>Potassium Level (mg/kg):</label>
        <input type="number" value={potassium} onChange={(e) => setPotassium(e.target.value)} />

        <label>Average Temperature (°C):</label>
        <input type="number" value={temperature} onChange={(e) => setTemperature(e.target.value)} />

        <label>Seasonal Rainfall (mm):</label>
        <input type="number" value={rainfall} onChange={(e) => setRainfall(e.target.value)} />

        <label>Humidity Level (%):</label>
        <input type="number" step="1" value={humidity} onChange={(e) => setHumidity(e.target.value)} />

        <label>Wind Conditions:</label>
        <select value={windConditions} onChange={(e) => setWindConditions(e.target.value)} >
          <option value="">Select...</option>
          <option value="Calm">Calm</option>
          <option value="Moderate">Moderate</option>
          <option value="Strong">Strong</option>
        </select>

        {/* Farming and Water Inputs */}
        <label>Water Source:</label>
        <select value={waterSource} onChange={(e) => setWaterSource(e.target.value)} >
          <option value="">Select...</option>
          <option value="Well">Well</option>
          <option value="River">River</option>
          <option value="Reservoir">Reservoir</option>
          <option value="None">None</option>
        </select>

        <label>Irrigation System Available:</label>
        <input
          type="checkbox"
          checked={irrigationSystem}
          onChange={(e) => setIrrigationSystem(e.target.checked)}
        /> Yes

        <label>Water Usage Limitations:</label>
        <select value={waterUsage} onChange={(e) => setWaterUsage(e.target.value)} >
          <option value="">Select...</option>
          <option value="None">None</option>
          <option value="Low">Low</option>
          <option value="Moderate">Moderate</option>
          <option value="High">High</option>
        </select>

        <label>Farming Technology Access:</label>
        <select value={farmingTech} onChange={(e) => setFarmingTech(e.target.value)} >
          <option value="">Select...</option>
          <option value="Mechanized">Mechanized</option>
          <option value="Automated">Automated</option>
          <option value="Manual">Manual</option>
        </select>

        <label>Previously Grown Crops:</label>
        <input type="text" value={prevCrops} onChange={(e) => setPrevCrops(e.target.value)} />

        <label>Crop Preferences:</label>
        <input type="text" value={cropPreferences} onChange={(e) => setCropPreferences(e.target.value)} />

        <label>Region/Place:</label>
        <input type="text" value={region} onChange={(e) => setRegion(e.target.value)} />

        <button type="submit" disabled={loading}>
          {loading ? "Generating..." : "Get Recommendations"}
        </button>
      </form>

      {/* Display Recommendations */}
      {recommendations && (
        <div className="recommendations">
          <h9>Crop Recommendations</h9>
          {recommendations.error ? (
            <p style={{ color: 'red' }}>{recommendations.error}</p>
          ) : (
            <ul>
              {recommendations.map((recommendation, index) => (
                <li key={index}>
                  <h4>{recommendation.crop}</h4>
                  <p><strong>Suitability:</strong> {recommendation.suitability}</p>
                  <p><strong>Expected Yield:</strong> {recommendation.expectedYield}</p>
                  <p><strong>Tips for Optimal Growth:</strong> {recommendation.tipsForOptimalGrowth}</p>
                  <p><strong>Environmental Considerations:</strong> {recommendation.environmentalConsiderations}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
export default CropRecommendation;