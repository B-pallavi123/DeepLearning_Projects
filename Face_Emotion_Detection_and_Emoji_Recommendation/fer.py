# -*- coding: utf-8 -*-
"""FER.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15H-rfYOZ-lkVOGZdENRrtH6voPE7_3j4

# **Importing Dataset**
"""

from google.colab import files
files.upload()  # Upload the downloaded `kaggle.json`

!pip install -q kaggle  # Install Kaggle API
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json  # Set permissions

!kaggle datasets download -d manishshah120/facial-expression-recog-image-ver-of-fercdataset

!unzip facial-expression-recog-image-ver-of-fercdataset.zip  # Unzip if needed

"""# **Importing Libraries**"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim

# Paths
DATA_DIR = "/content/Dataset/train"

# Parameters
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Dataset and Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])


dataset = ImageFolder(root=DATA_DIR, transform=transform)

"""# **Splitting the Dataset**"""

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""# **Building the Model**"""

# Model
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))  # Adjust for the number of classes
model = model.to(DEVICE)  # Move model to GPU/CPU

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU/CPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}")

# Validation Loop
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU/CPU
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

"""# **Trainig the Model**"""



# Main Training Process
for epoch in range(NUM_EPOCHS):
    train(model, train_loader, criterion, optimizer, epoch, DEVICE)
    validate(model, val_loader, criterion, DEVICE)

# Save the trained model
torch.save(model.state_dict(), "resnet_model.pth")
print("Model saved to resnet_model.pth")











































import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision import models
import torch.nn as nn

# Set up the app title and layout
st.set_page_config(page_title="Face Emotion Recognition", layout="wide")
st.title("Face Emotion Recognition with Emoji Recommendation")

# Define emotion-to-emoji mapping
emotion_emoji_mapping = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
}

# Define the emotion classes (adjust according to your model)
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load the pre-trained PyTorch model
@st.cache_resource
def load_emotion_model(model_path):
    # Load your custom model architecture (example uses ResNet)
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(emotion_classes))  # Adjust for your number of classes

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Face detection function
def detect_faces(image):
    # Convert to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Load face detector (Haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces, image_cv

# Image preprocessing for the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Main prediction function
def predict_emotion(model, image):
    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        emotion = emotion_classes[predicted.item()]

    return emotion

# Streamlit UI
def main():
    # Sidebar for model upload
    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.file_uploader("Upload your PyTorch model (.pth file)", type=["pth"])

    if model_path is not None:
        try:
            model = load_emotion_model(model_path)
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            return

    # Main content area
    st.header("Upload an image for emotion detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and 'model' in locals():
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Detect faces and emotions
        faces, image_cv = detect_faces(image)

        if len(faces) > 0:
            st.success(f"Detected {len(faces)} face(s) in the image")

            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                # Draw rectangle around face
                cv2.rectangle(image_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Crop face and predict emotion
                face_img = image.crop((x, y, x+w, y+h))
                emotion = predict_emotion(model, face_img)
                emoji = emotion_emoji_mapping.get(emotion, '❓')

                # Display results
                st.write(f"Face {i+1}:")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(face_img, caption=f"Face {i+1}", width=150)
                with col2:
                    st.subheader(f"Emotion: {emotion.capitalize()} {emoji}")
                    st.write(f"Recommended emoji: {emoji}")

            # Show image with face detection
            st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption='Face Detection', use_column_width=True)
        else:
            st.warning("No faces detected in the image. Please try another image.")

if __name__ == "__main__":
    main()

# Commented out IPython magic to ensure Python compatibility.
# %%writefile emotion_app.py
# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import cv2
# from torchvision import models
# import torch.nn as nn
# 
# # Set up the app
# st.set_page_config(page_title="Face Emotion Recognition", layout="wide")
# st.title("Face Emotion Recognition with Emoji Recommendation")
# 
# # Emotion to emoji mapping
# emotion_emoji_mapping = {
#     'angry': '😠',
#     'disgust': '🤢',
#     'fear': '😨',
#     'happy': '😊',
#     'sad': '😢',
#     'surprise': '😲',
#     'neutral': '😐'
# }
# 
# emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# 
# # Load model function
# @st.cache_resource
# def load_model(model_path):
#     model = models.resnet18(pretrained=False)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, len(emotion_classes))
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model
# 
# # Face detection
# def detect_faces(image):
#     image_cv = np.array(image)
#     image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     return faces, image_cv
# 
# # Image preprocessing
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0)
# 
# # Emotion prediction
# def predict_emotion(model, image):
#     input_tensor = preprocess_image(image)
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted = torch.max(output, 1)
#         return emotion_classes[predicted.item()]
# 
# def main():
#     # Model upload
#     st.sidebar.header("Model Configuration")
#     model_path = st.sidebar.file_uploader("Upload PyTorch model (.pth)", type=["pth"])
# 
#     if model_path:
#         try:
#             with open("temp_model.pth", "wb") as f:
#                 f.write(model_path.getbuffer())
#             model = load_model("temp_model.pth")
#             st.sidebar.success("Model loaded successfully!")
#         except Exception as e:
#             st.sidebar.error(f"Error loading model: {e}")
#             return
# 
#     # Image upload and processing
#     st.header("Upload an image for emotion detection")
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# 
#     if uploaded_file and 'model' in locals():
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
# 
#         faces, image_cv = detect_faces(image)
# 
#         if len(faces) > 0:
#             st.success(f"Found {len(faces)} face(s)")
# 
#             for i, (x, y, w, h) in enumerate(faces):
#                 cv2.rectangle(image_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                 face_img = image.crop((x, y, x+w, y+h))
#                 emotion = predict_emotion(model, face_img)
#                 emoji = emotion_emoji_mapping[emotion]
# 
#                 st.write(f"**Face {i+1}:** {emotion.capitalize()} {emoji}")
#                 st.image(face_img, width=150)
# 
#             st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption='Detected Faces')
#         else:
#             st.warning("No faces detected. Try another image.")
# 
# if __name__ == "__main__":
#     main()

# Commented out IPython magic to ensure Python compatibility.
# %%writefile emotion_app.py
# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import cv2
# from torchvision import models
# import torch.nn as nn
# 
# # Set up the app
# st.set_page_config(page_title="Face Emotion Recognition", layout="wide")
# st.title("Face Emotion Recognition with Emoji Recommendation")
# 
# # Emotion to emoji mapping
# emotion_emoji_mapping = {
#     'angry': '😠',
#     'disgust': '🤢',
#     'fear': '😨',
#     'happy': '😊',
#     'sad': '😢',
#     'surprise': '😲',
#     'neutral': '😐'
# }
# 
# emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# 
# # Load model function
# @st.cache_resource
# def load_model(model_path):
#     model = models.resnet18(pretrained=False)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, len(emotion_classes))
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model
# 
# # Face detection
# def detect_faces(image):
#     image_cv = np.array(image)
#     image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     return faces, image_cv
# 
# # Image preprocessing
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0)
# 
# # Emotion prediction
# def predict_emotion(model, image):
#     input_tensor = preprocess_image(image)
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted = torch.max(output, 1)
#         return emotion_classes[predicted.item()]
# 
# def main():
#     # Load model from fixed path
#     model_path = "/content/resnet_model.pth"  # Update this to your model's path
#     try:
#         model = load_model(model_path)
#         st.sidebar.success("Model loaded successfully!")
#     except Exception as e:
#         st.sidebar.error(f"Error loading model: {e}")
#         st.sidebar.error("Please ensure:")
#         st.sidebar.error("1. Model file exists at specified path")
#         st.sidebar.error("2. Model architecture matches expected format")
#         st.sidebar.error(f"Current path: {model_path}")
#         return
# 
#     # Image upload and processing
#     st.header("Upload an image for emotion detection")
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# 
#     if uploaded_file and 'model' in locals():
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
# 
#         faces, image_cv = detect_faces(image)
# 
#         if len(faces) > 0:
#             st.success(f"Found {len(faces)} face(s)")
# 
#             for i, (x, y, w, h) in enumerate(faces):
#                 cv2.rectangle(image_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                 face_img = image.crop((x, y, x+w, y+h))
#                 emotion = predict_emotion(model, face_img)
#                 emoji = emotion_emoji_mapping[emotion]
# 
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.image(face_img, width=200)
#                 with col2:
#                     st.subheader(f"Emotion: {emotion.capitalize()} {emoji}")
#                     st.write(f"Recommended emoji: {emoji}")
# 
#             st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption='Detected Faces', use_column_width=True)
#         else:
#             st.warning("No faces detected. Try another image.")
# 
# if __name__ == "__main__":
#     main()