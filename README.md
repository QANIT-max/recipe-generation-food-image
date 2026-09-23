# Recipe Generation Through Food Image

Upload a photograph of a dish and the app tells you what it is and how to cook it.
A Flask web application with a Keras image classifier behind it, built as a final year
project in BS Computer Science at The University of Lahore.

The idea came from a simple habit: people photograph food constantly, then still have to
search for the recipe by name. Here the photograph *is* the search.

---

## What it does

- A visitor registers or signs in (Firebase Authentication).
- They upload a photograph of a dish.
- An Xception convolutional neural network predicts which dish it is.
- The matching recipe — title, ingredients and steps — is returned with the image.

Routes: `/` (home), `/login`, `/logout`, `/find-my-recipe`, `/upload`, `/predict`.

## How a prediction actually works

1. `POST /predict` receives the file and opens it with Pillow, converting to RGB so PNGs
   with transparency do not break the tensor shape.
2. The image is resized to 224×224 and scaled to the 0–1 range the model was trained on.
3. `model.predict` returns probabilities across the classes; `argmax` picks the winner.
4. The class name is looked up in `recipie.json`, which holds the recipes, and the page
   renders the dish with its ingredients and instructions.
5. An index outside the known classes is rejected with a clear error instead of crashing.

## The model

| | |
|---|---|
| Architecture | Xception, transfer learning (Keras applications) |
| Dataset | [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — 75,750 training and 25,250 test images |
| Classes | 21 |
| Input | 256×256 during training, batch size 32 |
| Augmentation | rescale 1/255, shear 0.2, zoom 0.2, horizontal flip (`ImageDataGenerator`) |
| Training | 60 epochs, Adam optimiser, `ModelCheckpoint` and `CSVLogger` callbacks |
| Result | **89.9% accuracy** on the held-out test set (loss 0.35) |

The training notebook is `notebooks/food-classification.ipynb`, and the numbers above are
its own recorded output, not estimates.

## Repository layout

```
app.py                              the Flask application
recipie.json                        recipes: 20 dishes with ingredients and steps
requirements.txt                    what you need to run it
requirements-freeze.txt             the full pip freeze of the original environment
static/
  css/ images1/ js/ templates/      the site's styling, scripts and pages
notebooks/
  food-classification.ipynb         data preparation, training and evaluation
experiments/
  testing_flask.py                  an earlier Flask version, before authentication
  main.py, main_recipetest.py       Streamlit prototypes used while testing the model
  main_50class.py, test.py          a wider 50-class experiment
```

## Running it locally

```bash
python -m venv venv
venv\Scripts\activate          # macOS or Linux: source venv/bin/activate
pip install -r requirements.txt
```

Then three things the repository deliberately does not carry:

1. **The trained model.** `food21.keras` is about 245 MB, past GitHub's file limit. Train it
   with the notebook, or place your own file next to `app.py` and point `MODEL_PATH` at it.
2. **Firebase web config.** Copy `firebase_config.example.json` to `firebase_config.json`
   and fill in your Firebase project's values. Do the same for
   `static/js/config.example.js` → `static/js/config.js`.
3. **Firebase service account.** Download it from your Firebase project and save it as
   `credentials/firebase-adminsdk.json`, or set `FIREBASE_ADMIN_CREDENTIALS`. This file is a
   private key: it must never sit inside `static/`, which Flask serves to anyone.

```bash
python app.py        # http://127.0.0.1:5000
```

Environment variables: `FLASK_SECRET_KEY`, `MODEL_PATH`, `FIREBASE_CONFIG_FILE`,
`FIREBASE_ADMIN_CREDENTIALS`.

## Testing

Each part was covered with positive and negative cases — a valid photograph, a file that is
not an image, an unknown dish, a missing recipe entry, a signed-out visitor reaching the
upload page — and those cases were re-run after every change to the model or the routes.

## Known limits, honestly

- `recipie.json` covers 20 dishes, so a correct prediction outside that list has no recipe
  to show yet.
- The model sees 21 classes; anything else is guessed as the nearest one it knows.
- Training used 256×256 images while `/predict` resizes to 224×224. The network accepts it,
  but matching the two is the first thing to fix for better accuracy.
- A poor photograph — bad light, an odd angle, several dishes in frame — still misleads it.
- Recipes are global; regional and Pakistani dishes are the obvious next dataset.

## Team

Final year project, BS Computer Science, The University of Lahore (2025).
Built by **Qanit Ur Rehman** and **Nabeel**, supervised by Sanna Ullah Imtiaz.
