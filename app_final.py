import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

# Defining the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 37 * 37, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Load your pre-trained model
model = SimpleCNN()
model.load_state_dict(torch.load('bone_fracture_model_150.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define a function to preprocess the image
def preprocess_image(image):
    # Define the transformations: Convert to grayscale, resize and normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((150, 150)),  # Resize to the model's expected input shape
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Define the prediction function
def predict(image):
    processed_image = preprocess_image(image)
    with torch.no_grad():
        prediction = model(processed_image)
    return prediction.item()

# Streamlit app
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Bone Fracture Detection 🦴", "LinkedIn 🔗"])

if page == "Bone Fracture Detection 🦴":
    ironhack = Image.open('ironhack.png')
    st.image(ironhack, width=200)
    st.title('Bone Fracture Detection 🦴')
    st.write('Upload an X-ray image to detect if the bone is broken or not.')

    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', width=500)

        def alert_with_timeout(alert: str, timeout: int = 1):
            alert = st.warning(alert)
            time.sleep(timeout)
            alert.empty()  # hide alert
        
        alert_with_timeout("Classifying...")

        # Make a prediction
        prediction = predict(image)

        # Display the result
        if prediction < 0.5:  # Assuming the model outputs a single sigmoid value for binary classification
            ouch = Image.open('ouch.jpg')
            wasted = Image.open('wasted.jpg')
            st.image(wasted, width=200)
            st.write('The bone is **broken**. 🩹')
        else:
            thumbs = Image.open('thumbs.jpg')
            st.image(thumbs, width=200)
            st.write('The bone is **NOT broken**. ✅')

elif page == "LinkedIn 🔗":
    linkedin = Image.open('LinkedIn_icon.png')
    st.image(linkedin, width=200)
    st.title("LinkedIn 🔗")
    st.write("This application was created by the buddies:")
    
    # Creator 1
    st.subheader("Alexandre")
    st.write("[LinkedIn](https://www.linkedin.com/in/alex-conte/)")

    # Creator 2
    st.subheader("Lydia")
    st.write("[LinkedIn](https://www.linkedin.com/in/lylrg/)")

    # Creator 3
    st.subheader("Rodrigo")
    st.write("[LinkedIn](https://www.linkedin.com/in/rodrigo-pierini/)")
