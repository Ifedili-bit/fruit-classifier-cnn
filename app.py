import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
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
class_labels = {v: k for k, v in class_indices.items()}

# Streamlit UI
st.set_page_config(page_title="🍎 Fruit & Veg Classifier", layout="centered")
st.title("🍓 Fruit & Vegetable Classifier")
st.write("Upload an image, and the model will predict what fruit or vegetable it is!")

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1. Upload an image (jpg, jpeg, or png).  
    2. The image will be resized automatically.  
    3. The model predicts the class and shows top 3 probabilities.
    """
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Convert RGBA to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    prediction = model.predict(img_array)[0]
    
    # Get top 3 predictions
    top_indices = prediction.argsort()[-3:][::-1]
    top_probs = prediction[top_indices]
    
    # Display results
    st.subheader("Predictions")
    for i, idx in enumerate(top_indices):
        st.write(f"{i+1}. **{class_labels[idx]}** — {top_probs[i]*100:.2f}%")
