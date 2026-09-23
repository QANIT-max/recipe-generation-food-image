import os

from flask import Flask, request, redirect, jsonify, url_for, send_from_directory, render_template_string, session
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
from io import BytesIO
import base64
import firebase_admin
from firebase_admin import auth, credentials

app = Flask(__name__, static_folder='static')
# Set FLASK_SECRET_KEY in the environment before running anywhere but a laptop.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-not-for-production")

# Firebase web configuration. These values are safe in a browser, but they
# belong to a specific Firebase project, so they are not committed: copy
# firebase_config.example.json to firebase_config.json and fill in your own.
FIREBASE_CONFIG_FILE = os.environ.get("FIREBASE_CONFIG_FILE", "firebase_config.json")
try:
    with open(FIREBASE_CONFIG_FILE, "r") as f:
        firebase_config = json.load(f)
except Exception as e:
    print(f"Firebase web config not loaded from {FIREBASE_CONFIG_FILE}: {e}")
    firebase_config = {}

# Initialize Firebase Admin SDK. The service account file is a private key:
# keep it outside the repository and outside static/, which Flask serves.
ADMIN_CREDENTIALS = os.environ.get(
    "FIREBASE_ADMIN_CREDENTIALS", "credentials/firebase-adminsdk.json"
)
try:
    cred = credentials.Certificate(ADMIN_CREDENTIALS)
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase Admin SDK initialization error: {e}")

# Load the trained model. The file is about 245 MB, so it is not in the
# repository: see the README for how to train it or where to put it.
MODEL_PATH = os.environ.get("MODEL_PATH", "food21.keras")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Model loading error: {e}")

# Load the JSON file with recipes
try:
    with open('recipie.json', 'r') as f:  # Ensure the path is correct
        recipes_data = json.load(f)
except Exception as e:
    print(f"Error loading recipes JSON: {e}")
    recipes_data = {}

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
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Recipe Generation</title>
    <link rel="icon" href="{{ url_for('static', filename='images1/recipe.ico') }}" type="image/x-icon" />

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">

    <link rel="stylesheet" href="{{ url_for('static', filename='css/Home.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/FlipCard.css') }}">
                                  
    <style>
    /* Adjusted styling for the logo */
  .logo {
      width: 130px; /* Adjusted width */
      height: 80px; /* Adjusted height */
      background-image: url('{{ url_for('static', filename='images1/Recipelogo.png') }}');
      background-size: contain; /* Ensure the whole image fits */
      background-repeat: no-repeat; /* Prevent the image from repeating */
      background-position: center; /* Center the image within the div */
      animation: slide180 3s ease-in-out infinite; /* 3s for one full cycle, infinite loop */
  }
      /* Keyframes for horizontal sliding */
      @keyframes slide180 {
          0% {
              transform: translateX(0); /* Start at the initial position */
          }
          50% {
              transform: translateX(40px); /* Slide to the right by 40px */
          }
          100% {
              transform: translateX(0); /* Slide back to the initial position */
          }
      }
    </style>

  </head>
  <body>
    <!-- Navbar -->
    <nav
      class="navbar navbar-expand-lg navbar-light bg-light d-flex justify-content-between"
    >
      <div class="logo"></div> <!-- Animated logo added here -->
      <a
        style="color: rgb(0, 0, 73); font-weight: bold; font-size: 30px"
        class="navbar-brand mx-n1"
        href="#"
        >Recipe Generation from <b style="color: red">Food</b> Image </a
      >
      <form class="form-inline my-2 my-lg-0">
            <div >
        <a href="/find-my-recipe" class="mx-5 btn btn-primary my-2 my-sm-0"  style="background-color: #007bff; color: white; border: none; padding: 10px 20px; text-decoration: none; cursor: pointer; display: inline-block;">
           Find My Recipe</a>
            </div>
      </form>
    </nav>
    <!-- Carousel -->
    <div
      id="myCarousel"
      class="carousel slide"
      data-bs-ride="carousel"
      data-bs-interval="2000"
    >
      <div class="carousel-inner">
        <div class="carousel-item active">
          <img src="/static/images1/bg.jpg" class="d-block w-100" alt="..." />
          <div class="carousel-caption d-none d-md-block">
            <h5>"Recipe Generation"</h5>
            <p>
                  Instant Recipes from Your Favorite Dishes
            </p>
          </div>
        </div>
        <div class="carousel-item">
          <img src="./static/images1/bg1.jpg" class="d-block w-100" alt="..." />
          <div class="carousel-caption d-none d-md-block">
            <h5>"Fast & Accurate:"</h5>
            <p>
              Leverage cutting-edge AI technology to quickly identify food items.
            </p>
          </div>
        </div>
        <div class="carousel-item">
          <img src="/static/images1/checkbg.jpg" class="d-block w-100" alt="..." />
          <div class="carousel-caption d-none d-md-block">
            <h5>"Keep It Handy:"</h5>
            <p>
               Store the recipe on your device for easy access anytime.
            </p>
          </div>
        </div>
      </div>
      <button
        class="carousel-control-prev"
        type="button"
        data-bs-target="#myCarousel"
        data-bs-slide="prev"
      >
        <span class="carousel-control-prev-icon" aria-hidden="true"></span>
        <span class="visually-hidden">Previous</span>
      </button>
      <button
        class="carousel-control-next"
        type="button"
        data-bs-target="#myCarousel"
        data-bs-slide="next"
      >
        <span class="carousel-control-next-icon" aria-hidden="true"></span>
        <span class="visually-hidden">Next</span>
      </button>
    </div>
                                  
                                      <!-- Bootstrap JS and Popper.js -->
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.min.js"></script

    <!-- Main features -->
    <div>
      <h1
        style="
          font-family: 'Times New Roman', Times, serif;
          font-weight: bold;
          margin-top: 100px;
        "
        class="text-center text-black"
      >
        <u>Main Features</u>
      </h1>
    </div>

    <div class="container-fluid">
      <div class="row justify-content-center">
        <div class="col-lg-8">
          <div
            class="d-flex flex-wrap justify-content-md-between justify-content-center"
          >
            <div class="card text-center border border-0 mx-2 my-2">
              <div class="wrapper">
                <img src="/static/images1/card-1-1.png" class="cover-image" />
              </div>
              <h1
                class="title text-center"
                style="
                  font-family: 'Times New Roman', Times, serif;
                  font-weight: bold;
                "
              >
                Classification
              </h1>
              <img src="/static/images1/card-1-1.png" class="character" />
            </div>

            <div class="card text-center border border-0 mx-2 my-2">
              <div class="wrapper">
                <img src="static/images1/search.png" class="cover-image" />
              </div>
              <h1
                class="title text-center"
                style="
                  font-family: 'Times New Roman', Times, serif;
                  font-weight: bold;">
                Extraction
              </h1>
              <img src="/static/images1/search.png" class="character" />
            </div>

            <div class="card text-center border border-0 mx-2 my-2">
              <div class="wrapper">
                <img src="/static/images1/card-1.avif" class="cover-image" />
              </div>
              <h1
                class="title text-center"
                style="
                  font-family: 'Times New Roman', Times, serif;
                  font-weight: bold;
                "
              >
                Generation
              </h1>
              <img src="static/images1/card-1.png" class="character" />
            </div>
                                  
          
             <div class="card text-center border border-0 mx-2 my-2">
              <div class="wrapper">
                <img src="static/images1/search.png" class="cover-image" />
              </div>
              <h1
                class="title text-center"
                style="
                  font-family: 'Times New Roman', Times, serif;
                  font-weight: bold;
                "
              >
                Save Recipe
              </h1>
              <img src="/static/images1/search.png" class="character" />
            </div>
                                                              

          </div>
        </div>
      </div>
    </div>

    <!-- Overview Section  -->
    <div class="container" style="margin-top: 100px">
      <div class="row">
        <div class="col-md-5 flip">
          <!-- Image -->
          <img src="static/images1/side-image.png" class="img-fluid" alt="Image" />
        </div>
        <div class="col-md-6">
          <!-- Headings and Text -->
          <div style="margin-top: 50px">
            <h4 class="text-start" style="color: rgb(78, 78, 78)">Overview</h4>
            <h1
              class="text-start"
              style="
                font-family: 'Times New Roman', Times, serif;
                font-weight: bold;
                color: black;
              "
            >
            Recipe Generation from Food Image
            </h1>
            <p>
              The "Recipe Generation from Food Image" project is designed to simplify the process of finding recipes 
                for dishes seen in images. With the rise of social media and food photography, people often 
                come across appealing food images and wish to recreate those dishes at home. 
            </p>
            <p>
              <strong>Automate Recipe Generation:</strong> Provide a seamless process for users 
                to obtain recipes by simply uploading an image of the food. <br /><br />
              <strong>Improve User Experience:</strong> Enhance the cooking experience
                by making it easy to find and follow recipes for visually identified dishes.
              <br />
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Heading -->
    <div style="margin-top: 100px">
      <h1
        style="font-family: 'Times New Roman', Times, serif; font-weight: bold"
        class="text-center text-black my-3"
      >
        <u>About Us</u>
      </h1>
    </div>

    <!-- About Us -->
    <div
      style="padding: 100px"
      class="widget-container bg-black w-100 d-flex justify-content-evenly"
    >
      <div class="flip-box common-flip-style">
        <div
          style="background-image: url(/static/images1/vision.png)"
          class="box-front common-box-style"
        >
          <div class="box-content-wrapper"></div>
        </div>
        <div class="box-back common-box-style box-bgi-effect">
          <div class="box-content-wrapper">
            <div class="box-content">
              <h4 class="text-light">Our Vision</h4>
              <p style="font-size: 13px; color: white">
                Our vision is to revolutionize the culinary experience by seamlessly connecting visual recognition
                technology with culinary creativity, making it effortless for anyone to discover, prepare, 
                and enjoy delicious recipes inspired by their food images.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div class="flip-box common-flip-style">
        <div
          style="background-image: url(static/images1/mission2.png)"
          class="box-front common-box-style"
        >
          <div class="box-content-wrapper"></div>
        </div>
        <div class="box-back common-box-style box-bgi-effect">
          <div class="box-content-wrapper">
            <div class="box-content">
              <h3 class="text-light">Our Mission</h3>
              <p style="font-size: 13px; color: white">
                Our mission is to empower users to explore and create diverse culinary dishes by providing 
                an intuitive platform that generates accurate and personalized recipes from food images,
                leveraging advanced machine learning and AI technologies. We aim to inspire culinary exploration and 
                enhance the cooking experience for individuals at all skill levels.

Values
              </p>
            </div>
          </div>
        </div>
      </div>

      <div class="flip-box common-flip-style">
        <div
          style="background-image: url(/static/images1/values.png)"
          class="box-front common-box-style"
        >
          <div class="box-content-wrapper"></div>
        </div>
        <div class="box-back common-box-style box-bgi-effect">
          <div class="box-content-wrapper">
            <div class="box-content">
              <h3 class="text-light">Our Values</h3>
              <p style="font-size: 13px; color: white">
                <strong>Accuracy:</strong> We are committed to providing reliable and precise recipe recommendations.<br />
                <strong>Diversity:</strong>We celebrate culinary diversity, offering a wide range of recipes that cater to various cultures,
                 dietary preferences, and cooking styles.<br />
                <strong>User-Centricity:</strong> Harnessing the power of teamwork
                to advance cybersecurity through strategic partnerships.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer  -->
    <footer>
      <div class="container">
        <div class="row py-5">
          <div class="col-md-4 footer-column">
            <ul class="nav flex-column">
              <li class="nav-item">
                <span class="footer-title">Recipe Generation from Food Image</span>
              </li>
              <li class="nav-item my-2">
                <a href="#">Dashboard</a>
              </li>
              <li class="nav-item my-2">
                <a href="#">Home</a>
              </li>
              <li class="nav-item my-2">
                <a href="#">Services</a>
              </li>
              <li class="nav-item my-2">
                <a href="#">About Us</a>
              </li>
            </ul>
          </div>
          <div class="col-md-4 footer-column">
            <ul class="nav flex-column">
              <li class="nav-item">
                <span class="footer-title">Quick Links</span>
              </li>
              <li class="nav-item my-2">
                <a href="#">FAQ's</a>
              </li>
              <li class="nav-item my-2">
                <a href="#">News and articles</a>
              </li>
            </ul>
          </div>
          <div class="col-md-4 footer-column">
            <ul class="nav flex-column">
              <li class="nav-item">
                <span class="footer-title">Contact & Support</span>
              </li>
              <li class="nav-item my-2">
                <a href="#">+92 309 4105183</a>
              </li>
               <li class="nav-item my-2">
                <a href="#">+92 309 6956255</a>
              </li>
              <li class="nav-item my-2">
                <a href="#">Live chat</a>
              </li>
              <li class="nav-item my-2">
                <a href="#">Contact us</a>
              </li>
              <li class="nav-item my-2">
                <a href="#">Give feedback</a>
              </li>
            </ul>
          </div>
        </div>

        <div class="text-center"><i class="fas fa-ellipsis-h"></i></div>

        <div class="row text-center">
          <div class="col-md-4 box">
            <span class="copyright quick-links"
              >Copyright &copy; Recipe Generation from Food Image
              <script>
                document.write(new Date().getFullYear());
              </script>
            </span>
          </div>
          <div class="col-md-4 box">
            <ul class="list-inline social-buttons">
              <li class="list-inline-item">
                <a href="#">
                  <i class="fab fa-twitter"></i>
                </a>
              </li>
              <li class="list-inline-item">
                <a href="#">
                  <i class="fab fa-facebook-f"></i>
                </a>
              </li>
              <li class="list-inline-item">
                <a href="#">
                  <i class="fab fa-linkedin-in"></i>
                </a>
              </li>
            </ul>
          </div>
          <div class="col-md-4 box">
            <ul class="list-inline quick-links">
              <li class="list-inline-item">
                <a href="#">Privacy Policy</a>
              </li>
              <li class="list-inline-item">
                <a href="#">Terms of Use</a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </footer>                                  
</body>
</html>
    ''', firebase_config=firebase_config)

@app.route('/find-my-recipe')
def find_my_recipe():
    # Check if the user is logged in
    if 'user_id' in session:
        return redirect('/upload')
    else:
        return redirect(url_for('login', show_popup=True, message="Firstly, You should login/Signup to get the Recipe."))

@app.route('/login', methods=['GET', 'POST'])
def login():
    show_popup = request.args.get('show_popup', default=False, type=bool)
    alert_message = request.args.get('message', default="", type=str)

    if request.method == 'POST':
        # This part will now be handled on the frontend with Firebase Auth
        email = request.form.get('email')
        password = request.form.get('password')
        # Redirect to upload page after successful login
        return redirect('/upload')

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <title>Login & Registration Form</title>
          <link rel="icon" href="{{ url_for('static', filename='images1/recipe.ico') }}" type="image/x-icon" />

        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
        <script src="https://www.gstatic.com/firebasejs/8.10.1/firebase-app.js"></script>
        <script src="https://www.gstatic.com/firebasejs/8.10.1/firebase-auth.js"></script>
        <script src="https://www.gstatic.com/firebasejs/8.10.1/firebase-firestore.js"></script>
                                  
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            const firebaseConfig = {{ firebase_config|tojson }};
            firebase.initializeApp(firebaseConfig);
        </script>
        <style>
            .form {
                display: none;
            }
            .form.active {
                display: block;
            }
        </style>
    </head>
    <body>
         {% if alert_message %}
       <script>
            alert("{{ alert_message }}");
        </script>
        {% endif %}
                                  

        <div class="container">
            <div class="login form active">
                <header>Login</header>
                <form id="loginForm">
                    <input type="text" name="email" id='inUsr' placeholder="Enter your email" required>
                    <input type="password" name="password" id="inPass" placeholder="Enter your password" required>
                    <span class="signup"><a href="#" id="forgotLabel">Forgot password?</a></span>
                    <input type="submit" class="button loginbtn" value="Login">
                </form>
                <div class="signup">
                    <span class="signup">Don't have an account? <a href="#" id="signupLabel">Signup</a></span>
                </div>
            </div>

            <div class="registration form">
                <header>Signup</header>
                <form id="signupForm">
                    <input type="text" name="name" id="name" placeholder="Enter your name" required>
                    <input type="text" name="username" id='username' placeholder="Enter your username" required>
                    <input type="email" name="email" id='email' placeholder="Enter your email" required>
                    <input type="password" name="password" id='password' placeholder="Create a password" required>
                    <input type="submit" class="button signupbtn" value="Signup">
                </form>
                <div class="signup">
                    <span class="signup">Already have an account? <a href="#" id="loginLabel">Login</a></span>
                </div>
            </div>

            <div class="forgot form">
                <header>Forgot Password</header>
                <form id="forgotForm">
                    <input type="email" name="email" id='forgotinp' placeholder="Enter your email" required>
                    <input type="submit" class="button forgotbtn" value="Submit">
                </form>
                <div class="signup">
                    <span class="signup">Don't have an account? <a href="#" id="signupLabel">Signup</a></span>
                </div>
                <div class="signup">
                    <span class="signup">Already have an account? <a href="#" id="loginLabel">Login</a></span>
                </div>
            </div>
        </div>

        <script>
            // Show the modal if the show_popup flag is True
            const showPopup = {{ 'true' if show_popup else 'false' }};
            $(document).ready(function() {
                if (showPopup) {
                    $('#loginModal').modal('show');
                }
            });

            // Firebase Signup Logic
            document.getElementById('signupForm').addEventListener('submit', function (event) {
                event.preventDefault();
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;

                firebase.auth().createUserWithEmailAndPassword(email, password)
                    .then((userCredential) => {
                        const user = userCredential.user;

                        // Send email verification
                        user.sendEmailVerification()
                            .then(() => {
                                // Store the user ID in the session (optional, you might delay this until verification)
                                fetch('/set_session', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify({user_id: user.uid})
                                }).then(() => {
                                    alert("Signup successful! A verification email has been sent. Please verify your email before logging in.");
                                    window.location.href = "/login";
                                });
                            })
                            .catch((error) => {
                                alert("Error sending verification email: " + error.message);
                            });
                    })
                    .catch((error) => {
                        alert(error.message);
                    });
            });

   // Firebase Login Logic
document.getElementById('loginForm').addEventListener('submit', function (event) {
    event.preventDefault();
    const email = document.getElementById('inUsr').value;
    const password = document.getElementById('inPass').value;

    firebase.auth().signInWithEmailAndPassword(email, password)
        .then((userCredential) => {
            const user = userCredential.user;
            
            // Check if the user's email is verified
            if (user.emailVerified) {
                // Store the user ID in the session
                fetch('/set_session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({user_id: user.uid})
                }).then(() => {
                    alert("Login successful!");
                    window.location.href = "/upload";
                });
            } else {
                // Sign the user out if email is not verified
                firebase.auth().signOut().then(() => {
                    alert("Please verify your email before logging in.");
                });
            }
        })
        .catch((error) => {
            let errorMessage;
            switch (error.code) {
                case 'auth/wrong-password':
                    errorMessage = "Password is Incorrect, Please Enter the correct Password.";
                    break;
                case 'auth/user-not-found':
                    errorMessage = "No user found with this email. Please check your email or sign up.";
                    break;
                case 'auth/invalid-email':
                    errorMessage = "The email address is not valid. Please check the format.";
                    break;
                default:
                    errorMessage = error.message; // Generic message for other errors
                    break;
            }
            alert(errorMessage);
        });
});

            // Firebase Forgot Password Logic
            document.getElementById('forgotForm').addEventListener('submit', function (event) {
                event.preventDefault();
                const email = document.getElementById('forgotinp').value;

                firebase.auth().sendPasswordResetEmail(email)
                    .then(() => {
                        alert("Password reset email sent!");
                        window.location.href = "/login";
                    })
                    .catch((error) => {
                        alert(error.message);
                    });
            });

            // JavaScript to handle form switching
            document.getElementById('signupLabel').addEventListener('click', function() {
                document.querySelector('.login.form').classList.remove('active');
                document.querySelector('.forgot.form').classList.remove('active');
                document.querySelector('.registration.form').classList.add('active');
            });

            document.getElementById('loginLabel').addEventListener('click', function() {
                document.querySelector('.registration.form').classList.remove('active');
                document.querySelector('.forgot.form').classList.remove('active');
                document.querySelector('.login.form').classList.add('active');
            });

            document.getElementById('forgotLabel').addEventListener('click', function() {
                document.querySelector('.login.form').classList.remove('active');
                document.querySelector('.registration.form').classList.remove('active');
                document.querySelector('.forgot.form').classList.add('active');
            });
        </script>

        <script src="{{ url_for('static', filename='js/page0.js') }}" type="module"></script>
    </body>
    </html>
    ''',  alert_message=alert_message, firebase_config=firebase_config)

@app.route('/set_session', methods=['POST'])
def set_session():
    data = request.get_json()
    session['user_id'] = data['user_id']
    return jsonify({'status': 'success'})

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect('/login')
    
    return render_template_string('''
<!doctype html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Food Image</title>
    <link rel="icon" href="{{ url_for('static', filename='images1/recipe.ico') }}" type="image/x-icon" />

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/test.css') }}">
    <script src="https://www.gstatic.com/firebasejs/8.10.1/firebase-app.js"></script>
    <script src="https://www.gstatic.com/firebasejs/8.10.1/firebase-auth.js"></script>
    <script>
        const firebaseConfig = {{ firebase_config|tojson }};
        firebase.initializeApp(firebaseConfig);
    </script>
</head>

<body>
<div class="d-flex flex-column top-0 container-fluid mt-n5" style="height: 100vh;">           
    <nav class="navbar navbar-expand-lg navbar-light bg-light d-flex justify-content-between container-fluid mb-5">
      <a style="color: rgb(0, 0, 73); font-weight: bold; font-size: 30px" class="navbar-brand mx-3" href="#">
        Recipe Generation from <b style="color: red">Food</b> Image
      </a>
      <form class="form-inline my-2 my-lg-0">
         <a href="/" class="btn btn-primary mx-2">Home</a>
         <a href="/logout" class="btn btn-danger mx-2">Logout</a>
      </form>
    </nav>
                                  
    <div class="choose-file-wrapper mx-5 mt-5">
        <h1>Upload Food Image</h1>
        <div>
            <form action="/predict" method="post" enctype="multipart/form-data" onsubmit="return validateForm()">
                <div class="file-upload-container">
                    <input type="file" id="fileInput" name="file" accept="image/jpeg">
                    <span class="file-name">No file selected</span>
                    <button type="button" class="choose-file-button" onclick="document.getElementById('fileInput').click()">Choose file</button>
                </div>
                <button type="submit" class="upload-predict-button">Upload</button>
            </form>
        </div>
        <h5 class="text-secondary mt-3">Files Supported: JPG and PNG</h5>
      </div>
    </div>

    <script>
        function updateFileName() {
            const fileInput = document.getElementById('fileInput');
            const fileNameSpan = document.querySelector('.file-name');
            if (fileInput.files.length > 0) {
                fileNameSpan.textContent = fileInput.files[0].name;
            } else {
                fileNameSpan.textContent = "No file selected";
            }
        }

        function validateForm() {
            const fileInput = document.getElementById('fileInput');
            if (fileInput.files.length === 0) {
                alert("Please, Select a Food Image.");
                return false; // Prevent form submission
            }
            return true; // Allow form submission
        }

        document.getElementById('fileInput').addEventListener('change', updateFileName);
    </script>
    <script src="{{ url_for('static', filename='js/page1.js') }}" type="module"></script>
</body>
</html>
    ''', firebase_config=firebase_config)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))  # Resize the image to the size your model expects

    # Convert image to base64 for rendering in HTML
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  

    # Preprocess image as per your model requirements
    img_array = img_array / 255.0 

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0] - 1  

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

                <link rel="icon" href="{{ url_for('static', filename='images1/recipe.ico') }}" type="image/x-icon" />
                <link rel="stylesheet" href="{{{{ url_for('static', filename='css/test.css') }}}}">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.5.0-beta4/html2canvas.min.js"></script>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        color: #000;  /* Ensure text is black */
                        margin: 0;
                        padding: 0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        background-color: #333;  /* Set a darker background */
                    }}
                    .main-div{{
                        display: flex;
                        flex-direction: column;
                    }}
                    #capture {{
                        padding: 20px;
                        background: #555;  /* Darker background to improve contrast */
                        font-size: 14px;
                        line-height: 1.6;
                        max-width: 100%;
                        height: auto;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        color: #fff;  /* Set text color to white */
                    }}
                    .image-recipe-container {{
                        display: flex;
                        justify-content: space-between;
                        align-items: flex-start;
                    }}
                    .image-container {{
                        max-width: 50%;
                        margin-right: 20px;
                    }}
                    .recipe-container {{
                        max-width: 50%;
                    }}
                    .buttons {{
                        margin-top: 20px;
                        display: flex;
                        justify-content: flex-start;
                        gap: 10px;
                    }}
                    h1, h3 {{
                        color: #fff;  /* Make headers white */
                        font-weight: bold;
                        text-shadow: 1px 1px 2px #000;  /* Add shadow to text */
                    }}
                    h4 {{
                        color: #fff; /* White text for headings */
                        text-shadow: 1px 1px 2px #000;  /* Add shadow to text */
                    }}
                    ul, ol {{
                        color: #fff; /* White text for list items */
                        text-shadow: 1px 1px 2px #000;  /* Add shadow to text */
                    }}
                </style>
            </head>
            <body>

            <div class="d-flex flex-column top-0 container-fluid mt-n5 mt-n5" style="height: 100vh; ">           

            
            <nav
      class="navbar navbar-expand-lg navbar-light bg-light d-flex justify-content-between container-fluid mb-5 mb-5">
      <a
        style="color: rgb(0, 0, 73); font-weight: bold; font-size: 30px"
        class="navbar-brand mx-3"
        href="#"
        >Recipe Generation from <b style="color: red">Food</b> Image </a
      >
 <form class="form-inline my-2 my-lg-0">
    <a href="/upload" class="btn btn-secondary mx-2" style="background-color: black">Back</a>
</form>
    </nav>


            <div class="main-div">
                <div id="capture">
                    <div class="image-recipe-container">
                        <div class="image-container">
                            <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image" style="max-width:100%; height:auto;">
                        <h1 style="padding-top: 50px;">Predicted Food: {predicted_class_name}</h1>
                        </div>
                        <div class="recipe-container">
                            <div>{formatted_recipe}</div>
                        </div>
                    </div>
                </div>

                   <div class="buttons">
                        <button class="btn btn-primary" onclick="takescreenshot()">Save</button>
                    </div>
                </div>
                
                </div>
                <script>
                    function takescreenshot() {{
                        html2canvas(document.querySelector("#capture"), {{
                            scale: 2,  // Adjust scale for better clarity
                            useCORS: true // Use this to allow cross-origin images to be captured
                        }}).then(canvas => {{
                            var link = document.createElement('a');
                            link.download = 'recipe_screenshot.png';
                            link.href = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
                            link.click();
                        }});
                    }}
                </script>
            </body>
            </html>
            ''')
        else:
            return jsonify({'error': f'No recipe found for {predicted_class_name}'})
    else:
        return jsonify({'error': 'Invalid predicted class index.'})

if __name__ == '__main__':
    app.run(debug=True)
