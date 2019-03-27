from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template ,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/skin_model_1.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# # You can also use pretrained model from Keras
# # Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(75, 100) , grayscale=False)

    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
    
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        labels = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions', 'Basal cell carcinoma',
                          'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']

        labels = tuple(labels)

        # Make prediction
        preds = model_predict(file_path, model)

        preds = preds.tolist()

        # convert list of lists to one list for rounding to work
        flat_preds = [item for sublist in preds for item in sublist]

        updated_preds = list(
            map(lambda x: (round(x*100, 3)), flat_preds))

        dictionary = dict(zip(labels, updated_preds))

        # create a function which returns the value of a dictionary

        def keyfunction(k):
            return dictionary[k]

        global diagnosis
        diagnosis = []
      

        # sort by dictionary by the values and print top 3 {key, value} pairs
        for key in sorted(dictionary, key=keyfunction, reverse=True)[:3]:

            if dictionary[key] > 0:
                diagnosis.append([key, str(dictionary[key]) + "%"])

        
    return jsonify(diagnosis)
    

        


if __name__ == '__main__':
    app.run(port=5002, debug=True, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()