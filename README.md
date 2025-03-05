# 🌱 AgroHelp - Your Smart Agriculture Assistant

AgroHelp is a cutting-edge agriculture consultancy platform designed to empower farmers with expert guidance, real-time insights, and AI-driven solutions to boost productivity and sustainability.

##  Key Features
-  **Expert Consultation** – Connect with agricultural specialists for personalized guidance.
-  **AI-Powered Crop Selection** – Smart recommendations based on soil type, weather, and location.
-  **Live Weather Updates** – Get real-time weather insights for better decision-making.
-  **Market Price Tracker** – Stay updated on live crop prices.
-  **Farmer Community Forum** – Share experiences and solutions with fellow farmers.
-  **AI Disease Detection** – Detect plant diseases by uploading crop images.

##  Tech Stack
- **Frontend:** React.js
- **Backend:** Node.js, Express.js, Python (Flask for AI model integration)
- **Database:** MongoDB
- **Authentication:** JWT-based authentication
- **AI/ML:** AI-driven crop recommendations and disease detection

##  Project Structure
```
AgroHelp/
│── .git/                    # Version control
│── .vscode/                 # Editor configuration
│── AgroHelp/                # Main project directory
│── uploads/                 # Directory for image uploads
│── venv/                    # Virtual environment for dependencies
│── app.py                   # Backend for AI model processing
│── best_model.keras         # AI model (Keras) for disease detection
│── cmrit_model.pth          # AI model (PyTorch) for disease detection
│── README.md                # Project documentation
```

##  Getting Started
### 1️⃣ Installation
```sh
# Clone the repository
git clone <repository-url>

# Navigate to the project folder
cd AgroHelp

# Install frontend dependencies
npm install 

# Install Python backend dependencies
pip install -r requirements.txt
```
### 2️⃣ Running the Application
```sh
# Start the backend server
npm run server

# Start the frontend
npm start
```

##  How to Use
1️⃣ **Sign up or log in** to access the platform.
2️⃣ **Enter soil details** to receive AI-driven crop recommendations.
3️⃣ **Check live weather updates and market prices** to make informed farming decisions.
4️⃣ **Engage with experts or farmers** in the community forum.
5️⃣ **Upload crop images** for AI-powered disease detection.

##  Running the Disease Detection Model
Ensure the `uploads/` directory is created. Then, run:
```sh
python app.py
```
The backend will analyze the image and return disease predictions.

## 🔗 Access AI Disease Detection Models
Download the AI models from the links below:
- 📌 [Keras Model](https://drive.google.com/file/d/1rzkLE0V8QbIfexuF9cvYGf8n1Hw1LVc7/view?usp=drive_link)
- 📌 [PyTorch Model](https://drive.google.com/file/d/1j8u84LM6SEw_CN1tAjoyPi78mM0Ulc3R/view?usp=drive_link)

##  License
This project is licensed under the **MIT License**.
