import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model("fruit_classifier.h5")

# Class labels
class_indices = {
    'apple': 0, 'banana': 1, 'beetroot': 2, 'bell pepper': 3, 'cabbage': 4,
    'capsicum': 5, 'carrot': 6, 'cauliflower': 7, 'chilli pepper': 8, 'corn': 9,
    'cucumber': 10, 'eggplant': 11, 'garlic': 12, 'ginger': 13, 'grapes': 14,
    'jalepeno': 15, 'kiwi': 16, 'lemon': 17, 'lettuce': 18, 'mango': 19,
    'onion': 20, 'orange': 21, 'paprika': 22, 'pear': 23, 'peas': 24,
    'pineapple': 25, 'pomegranate': 26, 'potato': 27, 'raddish': 28,
    'soy beans': 29, 'spinach': 30, 'sweetcorn': 31, 'sweetpotato': 32,
    'tomato': 33, 'turnip': 34, 'watermelon': 35
}
# Reverse the dictionary
class_labels = {v: k for k, v in class_indices.items()}

# Streamlit UI
st.title("🍎 Fruit Classifier CNN")
st.write("Upload an image of a fruit or vegetable, and the model will predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and convert image to RGB (removes alpha channel if present)
    img = Image.open(uploaded_file).convert('RGB')
    
    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Shape: (1,224,224,3)
    
    # Make prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    st.success(f"Prediction: **{class_labels[class_idx]}**")
