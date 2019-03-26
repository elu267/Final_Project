from __future__ import division, print_function

# import necessary libraries
from flask import Flask, jsonify, render_template, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy

from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine


# coding=utf-8
import sys
import os
import glob
import re
import pandas as pd
import numpy as np

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras import backend as K


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None


def load_model1():
    global model
    global graph
    model = keras.models.load_model("models/skin_model_1.h5")
    graph = K.get_session().graph


load_model1()


def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    data = {"success": False}
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)

           # Load the saved image using Keras and resize it to the Xception
            # format of 299x299 pixels
            image_size = (75, 100)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size,
                                                    grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)

            global graph
            with graph.as_default():

                labels = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions', 'Basal cell carcinoma',
                          'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']

                preds = model.predict(image)
                # results = decode_predictions(preds)

                # result_df = pd.DataFrame(columns=['Label', "Prediction"])

                for x, y in zip(preds[0], labels):

                    updated_preds = list(
                        map(lambda x: str(round(x*100, 3)) + "%", preds[0]))

                    dictionary = dict(zip(labels, updated_preds))

                    # create a function which returns the value of a dictionary

                def keyfunction(k):
                    return dictionary[k]

            global diagnosis
            diagnosis = []

            # sort by dictionary by the values and print top 3 {key, value} pairs

            for key in sorted(dictionary, key=keyfunction, reverse=True)[:3]:
                print(key, dictionary[key])

                diagnosis.append([key, dictionary[key]])

    return jsonify(diagnosis)


if __name__ == "__main__":
    app.run(port=5002, debug=True, threaded=False)
    #  app.run(debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
