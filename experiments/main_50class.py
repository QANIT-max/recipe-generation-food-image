import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

# Cache the model loading
@st.cache_resource
def load_my_model():
    return load_model("food21_50classes(1).keras")

# Cache the JSON loading
@st.cache_data
def load_json_file():
    with open('recipie.json', 'r') as f:
        data = json.load(f)
    return data

# Load the model
model = load_my_model()

# Load the JSON file
recipes_data = load_json_file()

# Display the model summary in the Streamlit app
st.write("## Model Summary")
st.text(model.summary())

# Class names for decoding predictions
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla','chicken_wings','chocolate_cake',
    'chocolate_mousse','churros','clam_chowder','club_sandwich','crab_cakes','crab_cakes','crab_cakes','creme_brulee',
   'croque_madame','cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots','falafel',
   'filet_mignon','fish_and_chips','foie_gras','french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice',
   'frozen_yogurt','garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich'
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
    predicted_class = np.argmax(predictions, axis=1)[-1]  # Adjust index as necessary
    
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
