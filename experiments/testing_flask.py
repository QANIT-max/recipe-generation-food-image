from flask import Flask, request, jsonify, send_from_directory, render_template_string
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

app = Flask(__name__, static_folder='static')

# Load your Keras model
model = load_model("food21.keras")

# Load the JSON file with recipes
with open('recipie.json', 'r') as f:
    recipes_data = json.load(f)

# Class names for decoding predictions
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich'
]

@app.route('/')
def home():
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
    <title>Food Classification and Recipe Website</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" >  
                                
    </head>
    <body>
       <nav style="background-color: #f8f9fa; display: flex; justify-content: space-between; align-items: center; padding: 10px;">
    <a style="color: rgb(0, 0, 73); font-weight: bold; font-size: 30px; text-decoration: none;" href="#">Recipe Generation from <b style="color: red;">Food</b> Image</a>
    <form style="margin-right: 20px;">
       <a href="/upload" style="background-color: #007bff; color: white; border: none; padding: 10px 20px; text-decoration: none; cursor: pointer; display: inline-block;">
    For Food Recipe
</a>

    </form>
</nav>       

       <div id="carouselExampleControlsNoTouching" class="carousel slide" data-bs-touch="false">
  <div class="carousel-inner">
    <div class="carousel-item active">
      <img src="/static/images1/bg.jpg" class="d-block w-100" alt="...">
    </div>
    <div class="carousel-item">
      <img src="./static/images1/bg1.jpg" class="d-block w-100" alt="...">
    </div>
    <div class="carousel-item">
      <img src="./static/images1/checkbg.jpg" class="d-block w-100" alt="...">
    </div>
  </div>
  <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleControlsNoTouching" data-bs-slide="prev">
    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleControlsNoTouching" data-bs-slide="next">
    <span class="carousel-control-next-icon" aria-hidden="true"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>
    <h1 class="text-black">Upload Food Image <br> for Recipe</h1>


        
    
            <a href="/guideline">Guideline</a> |
            <a href="/upload">For Food Recipe</a>

    </body>
    </html>
    ''')

# @app.route('/guideline')
# def guideline():
#     return render_template_string('''
#     <!doctype html>
#     <html>
#     <head>
#         <title>Guideline</title>
#     </head>
#     <body>
#         <h1>Please Follow These Instructions</h1>
#         <ul>
#             <li>First, you should have a food image for the recipe.</li>
#             <li>The image should be in JPEG or JPG format.</li>
#             <li>Upload the food image, and you will receive a recipe.</li>
#             <li>Follow the recipe instructions to prepare the dish.</li>
#             <li>Experiment with the ingredients and instructions to suit your taste.</li>
#             <li>Share the recipes and your cooking experience with friends and family.</li>
#         </ul>
#         <h2>Enjoy Your Cooking!</h2>
#     </body>
#     </html>
#     ''')

@app.route('/upload')
def upload():
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Upload Food Image</title>
    </head>
    <body>
        <h1>Upload Food Image</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/jpeg">
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))  # Resize the image to the size your model expects
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Preprocess image as per your model requirements
    img = img / 255.0  # Example preprocessing

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Ensure the predicted class index is within the valid range
    if 0 <= predicted_class < len(class_names):
        predicted_class_name = class_names[predicted_class]

        # Generate recipe from JSON file
        if predicted_class_name in recipes_data:
            recipe_list = recipes_data[predicted_class_name]
            formatted_recipe = ""

            for recipe in recipe_list:
                title = recipe.get('title', 'No title')
                ingredients = recipe.get('ingredients', [])
                instructions = recipe.get('instructions', [])

                formatted_recipe += f"<h3>{title}</h3>"
                formatted_recipe += "<h4>Ingredients:</h4>"
                formatted_recipe += "<ul>"
                for ingredient in ingredients:
                    formatted_recipe += f"<li>{ingredient}</li>"
                formatted_recipe += "</ul>"
                formatted_recipe += "<h4>Instructions:</h4>"
                formatted_recipe += "<ol>"
                for instruction in instructions:
                    formatted_recipe += f"<li>{instruction}</li>"
                formatted_recipe += "</ol>"
                formatted_recipe += "<hr>"

            return render_template_string(f'''
            <!doctype html>
            <html>
            <head>
                <title>Recipe for {predicted_class_name}</title>
            </head>
            <body>
                <h1>Predicted Food: {predicted_class_name}</h1>
                <div>{formatted_recipe}</div>
                <a href="/upload">Upload Another Image</a>
            </body>
            </html>
            ''')
        else:
            return jsonify({'error': f'No recipe found for {predicted_class_name}'})
    else:
        return jsonify({'error': 'Invalid predicted class index.'})

if __name__ == '__main__':
    app.run(debug=True)
