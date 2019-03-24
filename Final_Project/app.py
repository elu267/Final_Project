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

# Keras
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from keras.applications.resnet50 import ResNet50

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None

def load_model():
    global model
    global graph
    model = keras.models.load_model("models/skin_model_1.h5")
    graph = K.get_session().graph


load_model()

def prepare_image(img):
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    print(img)
    # # Scale from 0 to 255
    # img /= 255
    # # Invert the pixels
    # img = 1 - img
    # # Flatten the image to an array of pixels
    # image_array = img.flatten().reshape(-1, 28 * 28)
    img = np.arange(27500).reshape(100,75)
    img = img.reshape((img.shape[0]*75, 100,3))
    # Return the processed feature array
    return image_array


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

            # Load the saved image using Keras and resize it to the mnist
            # format of 28x28 pixels
            image_size = (275, 100, 3)
            im = image.load_img(filepath, target_size=image_size,
                                grayscale=True)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(im)
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                predicted_digit = model.predict_classes(image_array)[0]
                data["prediction"] = str(predicted_digit)

                # indicate that the request was a success
                data["success"] = True

            return jsonify(data)
        
    return None


if __name__ == "__main__":
    app.run(port=5002, debug=False, threaded=False)
    #  app.run(debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
