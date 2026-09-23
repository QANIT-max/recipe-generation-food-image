import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

# Cache the model loading
@st.cache_resource
def load_my_model():
    return load_model("food21_update.keras")

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

# Class names for decoding predictions
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
    'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
    'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel',
    'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich'
]

# Define the navbar
st.sidebar.title("Navigation")
navbar = st.sidebar.radio('Go to', ['Home', 'Guideline', 'For Food Recipe'])

# Custom CSS for white text color and other styling
st.markdown(
    """
    <style>
    .recipe-text {
        color: white;
        white-space: pre-wrap;
    }
    .navbar {
        display: flex;
        justify-content: space-between;
        background-color: #4CAF50;
        padding: 14px 16px;
    }
    .navbar a {
        color: white;
        padding: 14px 16px;
        text-decoration: none;
        text-align: center;
    }
     .navbar a:hover {
        background-color: #111;
    }
    /* Sidebar */
    sidebar {
       margin-top:: 100px;
    }
    .css-1lcbmhc .css-1hynsf2 {
        color: white !important;
    }
    .css-1lcbmhc .css-1hynsf2 a {
        color: white !important;
    }
    .css-1lcbmhc .css-1hynsf2 a:hover {
        background-color: #111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# # Horizontal navigation bar
# st.markdown(
#     """
#    <ul class="navbar">
#         <li><a href="?nav=Home">Home</a></li>
#         <li><a href="?nav=Guideline">Guideline</a></li>
#         <li><a href="?nav=For food Recipe">For Food Recipe</a></li>
#     </ul>
#     """,
#     unsafe_allow_html=True
# )

# Main content
if navbar == 'Home':
    st.write("# Welcome to the Food Classification and Recipe Website!")
    st.write("Here you can upload a food image for a recipe.")

elif navbar == 'Guideline':
    st.write("## Please Follow These Instructions")
    st.write("#### 1. First, you should have a food image for the recipe.")
    st.write("#### 2. The image should be in JPEG or JPG format.")
    st.write("#### 3. Upload the food image, and you will receive a recipe.")
    st.write("#### 4. Follow the recipe instructions to prepare the dish.")
    st.write("#### 5. Experiment with the ingredients and instructions to suit your taste.")
    st.write("#### 6. Share the recipes and your cooking experience with friends and family.")
    st.write("## Enjoy Your Cooking!")

elif navbar == 'For Food Recipe':
    st.write("## Upload Food Image")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Load and preprocess the image
        img = load_img(uploaded_file, target_size=(224, 224))  # Adjust size as per your model's input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale the image array

        st.image(img, caption='Uploaded Food Image for Recipe.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make a prediction
        predictions = model.predict(img_array)
        st.write(f"Predictions: {predictions}")

        # Decode the predictions
        predicted_class = np.argmax(predictions, axis=1) -1  # Adjust class index

        # Ensure the predicted class index is within the valid range
        if 0 <= predicted_class[0] < len(class_names):
            predicted_class_name = class_names[predicted_class[0]]
            st.write(f'Predicted Food: {predicted_class_name}')

            # Generate recipe from JSON file
            if predicted_class_name in recipes_data:
                recipe_list = recipes_data[predicted_class_name]
                formatted_recipe = ""

                for recipe in recipe_list:
                    title = recipe.get('title', 'No title')
                    ingredients = recipe.get('ingredients', [])
                    instructions = recipe.get('instructions', [])

                    formatted_recipe += f"**Title:** {title}\n\n"
                    formatted_recipe += "**Ingredients:**\n"
                    formatted_recipe += "\n".join(ingredients) + "\n\n"
                    formatted_recipe += "**Instructions:**\n"
                    formatted_recipe += "\n".join(instructions) + "\n\n"
                    formatted_recipe += "---\n\n"

                st.markdown(f"<div class='recipe-text'>{formatted_recipe}</div>", unsafe_allow_html=True)
            else:
                st.write(f"No recipe found for {predicted_class_name}")
        else:
            st.write("Invalid predicted class index.")
