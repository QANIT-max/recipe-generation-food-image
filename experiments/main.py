import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model = load_model("food21_update.keras")

# Display the model summary in the Streamlit app
st.write("## Model Summary")
st.text(model.summary())

# Class names for decoding predictions
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla'
]

st.write("## Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

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
    predicted_class = np.argmax(predictions, axis=1) - 1  # Adjust index as necessary
    
    # Print the predicted class
    st.write(f'Predicted class: {class_names[predicted_class[0]]}')


