import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision import models
import torch.nn as nn

# Set up the app
st.set_page_config(page_title="Face Emotion Recognition", layout="wide")
st.title("Face Emotion Recognition with Emoji Recommendation")

# Emotion to emoji mapping
emotion_emoji_mapping = {
    'anger': '😠',
    'disgust': '🤢', 
    'fear': '😨',
    'happiness': '😊',
    'sadness': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

emotion_classes = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

# Load model function
@st.cache_resource
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(emotion_classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Face detection
def detect_faces(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces, image_cv

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Emotion prediction
def predict_emotion(model, image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return emotion_classes[predicted.item()]

def main():
    # Load model from fixed path
    model_path = r"C:\Users\RGUKT\Downloads\resnet_model.pth"  # Update this to your model's path 
    try:
        model = load_model(model_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.sidebar.error("Please ensure:")
        st.sidebar.error("1. Model file exists at specified path")
        st.sidebar.error("2. Model architecture matches expected format")
        st.sidebar.error(f"Current path: {model_path}")
        return
    
    # Image upload and processing
    st.header("Upload an image for emotion detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and 'model' in locals():
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        faces, image_cv = detect_faces(image)
        
        if len(faces) > 0:
            st.success(f"Found {len(faces)} face(s)")
            
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(image_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_img = image.crop((x, y, x+w, y+h))
                emotion = predict_emotion(model, face_img)
                emoji = emotion_emoji_mapping[emotion]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(face_img, width=200)
                with col2:
                    st.subheader(f"Emotion: {emotion.capitalize()} {emoji}")
                    st.write(f"Recommended emoji: {emoji}")
            
            st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption='Detected Faces', use_column_width=True)
        else:
            st.warning("No faces detected. Try another image.")

if __name__ == "__main__":
    main()
