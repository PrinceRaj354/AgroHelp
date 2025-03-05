"# plant-disease-detection" 
# AgroHelp

AgroHelp is an agriculture consultant platform designed to assist farmers with expert advice, real-time data, and resource management to improve productivity and sustainability.

## Features
- **Expert Consultation**: Connect with agricultural experts for personalized guidance.
- **Crop Recommendations**: AI-based crop selection based on soil type, weather, and location.
- **Weather Insights**: Real-time weather updates for better farming decisions.
- **Market Prices**: Live market price tracking for various crops.
- **Community Forum**: A space for farmers to share experiences and solutions.
- **Disease Detection**: Upload images of crops to detect diseases using AI models.

## Tech Stack
- **Frontend**: React.js
- **Backend**: Node.js, Express.js, Python (Flask for AI model integration)
- **Database**: MongoDB
- **Authentication**: JWT-based authentication
- **AI & ML**: AI models for crop prediction and disease detection

## Project Structure
```
AgroHelp/
│── .git/
│── .vscode/
│── AgroHelp/                # Main project folder
│── uploads/                 # Directory for storing uploaded images
│── venv/                    # Virtual environment for Python dependencies
│── app.py                   # Backend script for AI model inference
│── best_model.keras         # Keras model for disease detection
│── cmrit_model.pth          # PyTorch model for disease detection
│── README.md                # Project documentation
```

## Installation
```bash
# Clone the repository
git clone https://github.com/PrinceRaj354/AgroHelp.git

# Navigate to project directory
cd AgroHelp

# Install dependencies for frontend and backend
npm install  # For frontend
pip install -r requirements.txt  # For Python backend

# Start the backend server
npm run server

# Start the frontend
npm start
```

## Usage
1. Sign up or log in to access features.
2. Enter soil details for AI-driven crop recommendations.
3. View weather updates and market prices.
4. Consult experts or participate in the community forum.
5. Upload an image of your crop to detect diseases using AI models.

## Running the Disease Detection Model
- Ensure the `uploads/` directory exists for image uploads.
- Run the AI model backend using:
```bash
python app.py
```
- The API will process the uploaded image and return disease predictions.

## Accessing the Disease Detection Model
To access the AI model for disease detection, follow this [Google Drive link](#) (replace with actual link).

## Contributing
Contributions are welcome! Feel free to fork the repo, make changes, and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For queries, reach out at [your email] or visit our website at [project website].
