

# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms
# from PIL import Image
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Enable CORS for React app running at http://localhost:3000
# CORS(app)

# # Ensure uploads folder exists
# if not os.path.exists('uploads'):
#     os.makedirs('uploads')

# # Define the PlantDiseaseClassifier model (as per your architecture)
# class PlantDiseaseClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(PlantDiseaseClassifier, self).__init__()
#         self.resnet = models.resnet50(pretrained=True)
#         for param in self.resnet.parameters():
#             param.requires_grad = False
#         for param in self.resnet.layer4.parameters():
#             param.requires_grad = True
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.6),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         return self.resnet(x)

# # Load the trained model
# model = PlantDiseaseClassifier(num_classes=38)  # Adjust num_classes to match your setup
# model.load_state_dict(torch.load('cmrit_model.pth', map_location=torch.device('cpu')))
# model.eval()  # Set the model to evaluation mode

# # Class labels
# class_labels = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
#     'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 
#     'Cherry___healthy', 'Corn___Cercospora_leaf_spot', 'Corn___healthy', 
#     'Grape___Black_rot', 'Grape___Esca', 'Grape___healthy', 'Grape___Leaf_blight', 
#     'Lemon___Greening', 'Lemon___healthy', 'Peach___Bacterial_spot', 'Peach___healthy', 
#     'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
#     'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Strawberry___Leaf_scorch', 
#     'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
#     'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot', 
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy', 'Apple___Black_rot', 
#     'Apple___Cedar_apple_rust', 'Tomato___healthy'
# ]

# # Image preprocessing function
# def preprocess_image(image_path, img_height=256, img_width=256):
#     img = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.Resize((img_height, img_width)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img_tensor = transform(img)
#     img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
#     return img_tensor

# # API endpoint for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400

#     image_file = request.files['image']
    
#     if not image_file.content_type.startswith('image/'):
#         return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

#     # Save the uploaded image
#     image_path = os.path.join('uploads', image_file.filename)
#     image_file.save(image_path)

#     # Preprocess the image
#     processed_image = preprocess_image(image_path)

#     # Predict the class
#     with torch.no_grad():
#         output = model(processed_image)
#         _, predicted_class_index = torch.max(output, 1)

#     predicted_class_label = class_labels[predicted_class_index.item()]
#     predicted_class_probability = torch.softmax(output, dim=1)[0][predicted_class_index.item()].item()

#     # Return prediction with image URL and probability
#     return jsonify({
#         'predicted_class': predicted_class_label,
#         'probability': float(predicted_class_probability),
#         'image_url': f'http://localhost:5000/uploads/{image_file.filename}'
#     })

# # Route to serve uploaded images
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory('uploads', filename)

# if __name__ == '__main__':
#     app.run(debug=True)



































from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for React app running at http://localhost:3000
CORS(app)

# Ensure uploads folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Define the PlantDiseaseClassifier model
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Load the trained model
model = PlantDiseaseClassifier(num_classes=38)  # Adjust num_classes to match your setup
model.load_state_dict(torch.load('cmrit_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Class labels
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 
    'Cherry___healthy', 'Corn___Cercospora_leaf_spot', 'Corn___healthy', 
    'Grape___Black_rot', 'Grape___Esca', 'Grape___healthy', 'Grape___Leaf_blight', 
    'Lemon___Greening', 'Lemon___healthy', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy', 'Tomato___Mosaic_Virus', 
    'Tomato___Tomato_Yellow_Leaf_Curl', 'Pepper___Powdery_mildew', 'Pepper___Leaf_spot'
]


# Image preprocessing function
def preprocess_image(image_path, img_height=256, img_width=256):
    img = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    transform = transforms.Compose([
        # Adjust brightness and contrast
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # Resize the image
        transforms.Resize((img_height + 32, img_width + 32)),  # Resize slightly larger
        # Random crop to introduce variations
        transforms.CenterCrop((img_height, img_width)),
        # Random rotation to handle slight angular misalignments
        transforms.RandomRotation(degrees=15),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize to match ResNet training conditions
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Optionally, add Gaussian noise
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    
    if not image_file.content_type.startswith('image/'):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

    # Save the uploaded image
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Predict the class
    with torch.no_grad():
        output = model(processed_image)
        softmax_output = torch.softmax(output, dim=1)[0]
        top_5_probs, top_5_indices = torch.topk(softmax_output, k=5)

    # Get top 5 predictions
    top_predictions = [
        {
            'class': class_labels[idx],
            'probability': prob.item()
        } 
        for idx, prob in zip(top_5_indices, top_5_probs)
    ]

    # Define a threshold for confidence
    confidence_threshold = 0.7  # Adjust this value as needed

    # Check if the top prediction meets the confidence threshold
    top_prediction = top_predictions[0]
    if top_prediction['probability'] < confidence_threshold:
        return jsonify({
            'error': f'Low confidence prediction. Model is uncertain about the image.',
            'top_predictions': top_predictions,
            'image_url': f'http://localhost:5000/uploads/{image_file.filename}'
        }), 200

    # Return prediction with image URL and top predictions
    return jsonify({
        'predicted_class': top_prediction['class'],
        'probability': top_prediction['probability'],
        'top_predictions': top_predictions,
        'image_url': f'http://localhost:5000/uploads/{image_file.filename}'
    })

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)