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
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50

app = Flask(__name__)

# Bills base testing code.


def billsSkiResort(): {

    #################################################
    # Database Setup
    #################################################

    # app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///../Resources/clean_finalProjectL2.sqlite"
    # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # db = SQLAlchemy(app)

    # # reflect an existing database into a new model
    # Base = automap_base()
    # # reflect the tables
    # Base.prepare(db.engine, reflect=True)

    # # Save references to each table
    # finalProjectL2 = Base.classes.clean_finalProjectL2

    # # create route that renders index.html template
    # @app.route("/")
    # def index():
    #    """Return the homepage."""
    #    print("reading the index function")
    #    return render_template("indexTEST.html")

    # @app.route("/metadata")
    # def meta():
    #    """Go To Meta data data page"""
    #    print("reading the Meta data function")
    #    return render_template("Meta.html")

    # @app.route("/lesionLegion")
    # def leasion():
    #    stmt = db.session.query(finalProjectL2).statement
    #    df = pd.read_sql_query(stmt, db.session.bind)

    #    data = df.to_dict('records')

    #    return jsonify(data)

}


# Model saved with Keras model.save()
MODEL_PATH = 'models/skin_model_1.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


# global model
# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')
# global graph
# graph = tf.get_default_graph()


# First image processing provided by base code

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


# second image processing test

# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(75, 100))

#     # add a global spatial average pooling layer
#     x = MODEL_PATH.output
#     x = GlobalAveragePooling2D()(x)
#     # add a fully-connected layer
#     x = Dense(1024, activation='relu')(x)
#     # and a logistic layer -- let's say we have 7 classes
#     predictions = Dense(7, activation='softmax')(x)
#     model = Model(inputs=MODEL_PATH.input, outputs=predictions)

#     preds = model.predict(x)
#     return preds


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

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == "__main__":
    app.run(port=5002, debug=True)
    #  app.run(debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
