# import necessary libraries
from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import os

import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

app = Flask(__name__)

#################################################
# Database Setup
#################################################

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///../Resources/clean_finalProjectL2.sqlite"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)

# Save references to each table
finalProjectL2 = Base.classes.clean_finalProjectL2

# create route that renders index.html template
@app.route("/")
def index():
    """Return the homepage."""
    print("reading the index function")
    return render_template("indexTEST.html")

@app.route("/metadata")
def meta():
   """Go To Meta data data page"""
   print("reading the Meta data function")
   return render_template("Meta.html")

@app.route("/lesionLegion")
def leasion():
    stmt = db.session.query(finalProjectL2).statement
    df = pd.read_sql_query(stmt, db.session.bind)

    data = df.to_dict('records')

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)