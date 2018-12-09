"""
This code creates a small flask app which takes a link to an image as an input
and classifies the images as a dog or a cat. It uses a network and existing
weights to classify the images.
"""

from flask import Flask, render_template, request
from skimage import io
import os

from our_keras_model import load_from_url, create_model

IMG_SIZE = 64

os.environ['KMP_DUPLICATE_LIB_OK']='True'
app = Flask(__name__)

# We instantiate the model once
web_classifier = create_model()
# Not using this causes an error (https://github.com/keras-team/keras/issues/6462)
web_classifier._make_predict_function()
web_classifier.load_weights("Model-Checkpoints/model-checkpoints2")


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    img = load_from_url(text)
    result = web_classifier.predict_classes(img)
    if result == [[1]]:
        classified_as = "Dog"
    else:
        classified_as = "Cat"
    return render_template("classifier-response.html", classification=classified_as)


if __name__ == "__main__":
    app.run(debug=True)
