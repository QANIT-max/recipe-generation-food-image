import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import streamlit.components.v1 as components

@st.cache_resource
def load_my_model():
    return load_model("food21_update.keras")

@st.cache_data
def load_json_file():
    with open('recipie.json', 'r') as f:
        data = json.load(f)
    return data

model = load_my_model()
recipes_data = load_json_file()

st.write("## Model Summary")
st.text(model.summary())

class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla'
]

# Render the HTML template
components.html(open('Html\\Home.html').read(), height=600)

# Render CSS
st.markdown(f'<style>{open("CSS/Home.css").read()}</style>', unsafe_allow_html=True)

st.write("## Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="uploader")

if uploaded_file is not None:
    # Load and preprocess the image
    img = load_img(uploaded_file, target_size=(224, 224))  # Adjust size as per your model's input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image array

    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make a prediction
    predictions = model.predict(img_array)
    st.write(f"Predictions: {predictions}")

    # Decode the predictions
    predicted_class = np.argmax(predictions, axis=1 )-1,

    # Print the predicted class
    predicted_class_name = class_names[predicted_class]
    st.write(f'Predicted class: {predicted_class_name}')

    # Generate recipe from JSON file
    if predicted_class_name in recipes_data:
        recipe = recipes_data[predicted_class_name]
        st.write(f"## Recipe for {predicted_class_name}")
        st.write(recipe)
    else:
        st.write(f"No recipe found for {predicted_class_name}")
