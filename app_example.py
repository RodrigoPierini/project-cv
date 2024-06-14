import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # or import torch if you're using PyTorch
import time

# Load pre-trained model
model = tf.keras.models.load_model('model_converted_new_final.keras')  # Adjust if using PyTorch

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')  # Convert to grayscale if model expects single channel
    image = image.resize((100, 100))  # Resize to the model's expected input shape
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the prediction function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

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
