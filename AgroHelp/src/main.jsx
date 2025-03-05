import { StrictMode, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';
import Header from './components/custom/Header.jsx';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
import { LanguageProvider } from './languages/LanguageContext';
import CropRecommendation from './crop-recomment/index.jsx';
import FertilizerRecommendation from './fertilizer-suggestion/index.jsx';
import PlantDiseaseDetector from './plant-disease-detection/index.jsx';
import Viewtrip from './view-trip/[tripId]/index.jsx';

// Function to load Google Translate script
const loadGoogleTranslateScript = () => {
  const script = document.createElement('script');
  script.src = 'https://translate.google.com/translate_a/element.js?cb=loadgoogleTranslate';
  script.async = true;
  document.body.appendChild(script);
  
  window.loadgoogleTranslate = function () {
    new window.google.translate.TranslateElement(
      {
        pageLanguage: 'en', // Default language
        includedLanguages: 'kn,en,hi', // Allowed languages (Kannada, English, Hindi)
        layout: google.translate.TranslateElement.InlineLayout.SIMPLE, // Inline layout for better integration
      },
      'google_element'
    );
  };
};

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
  },
  {
    path: '/crop-recommendation',
    element: <CropRecommendation />,
  },
  {
    path: '/fertilizer-suggestion',
    element: <FertilizerRecommendation />,
  },
  {
    path: '/plant-disease-detection',
    element: <PlantDiseaseDetector />,
  },
  {
    path: '/view-trip/:tripId',
    element: <Viewtrip />,
  },
]);

const MainApp = () => {
  useEffect(() => {
    loadGoogleTranslateScript();
  }, []); // Runs once when the component mounts

  return (
    <StrictMode>
      <LanguageProvider>
        <Header />
        <div id="google_element" style={{ position: 'absolute', top: '10px', right: '10px' }}></div> {/* Google Translate Element */}
        <RouterProvider router={router} />
      </LanguageProvider>
    </StrictMode>
  );
};




createRoot(document.getElementById('root')).render(<MainApp />);
