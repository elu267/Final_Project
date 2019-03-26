# Final_Project

## Objective
Create a web application that allows one to investigate skin abnormalities such as moles / lesions and determine whether they should visit with a doctor or dermatologist for further testing. 

## Overview
This the HAM10000 ("Human Against Machine with 10000 training images") dataset. It consists of 10,015 dermatoscopicimages which are released as a training set for academic machine learning purposes and are publiclyavailable through the ISIC archive. 

#### Leision Types
##### nv:
Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants, which all are included in our series. The variants may differ significantly from a dermatoscopic point of view.

##### mel:
Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants. If excised in an early stage it can be cured by simple surgical excision. Melanomas can be invasive or non-invasive (in situ). We included all variants of melanoma including melanoma in situ, but did exclude non-pigmented, subungual, ocular or mucosal melanoma.

##### bkl:
"Benign keratosis" is a generic class that includes seborrheic ker- atoses ("senile wart"), solar lentigo - which can be regarded a flat variant of seborrheic keratosis - and lichen-planus like keratoses (LPLK), which corresponds to a seborrheic keratosis or a solar lentigo with inflammation and regression.

##### bcc:
Basal cell carcinoma is a common variant of epithelial skin cancer that rarely metastasizes but grows destructively if untreated. It appears in different morphologic variants (flat, nodular, pigmented, cystic, etc), which are all included in this set.

##### akiec:
Actinic Keratoses (Solar Keratoses) and intraepithelial Carcinoma (Bowen’s disease) are common non-invasive, variants of squamous cell car- cinoma that can be treated locally without surgery. Some authors regard them as precursors of squamous cell carcinomas and not as actual carci- nomas.

##### vasc:
Vascular skin lesions in the dataset range from cherry angiomas to angiokeratomas and pyogenic granulomas. Hemorrhage is also included in this category.

##### df:
Dermatofibroma is a benign skin lesion regarded as either a benign proliferation or an inflammatory reaction to minimal trauma. It is brown often showing a central zone of fibrosis dermatoscopically.

## Data Preparation

### Step 1: Importing Libraries
```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, DepthwiseConv2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
```

### Step 2: Data Exploration and Cleanup
Check for missing information `<skin_df.isnull().sum()>`. Only the age column has missing values; therefore, fill in missing values `<skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)>`. 

Exploring the features of the data set:
![Types of Lesions](Final_Project/static/images/lesionTypes.png)
![Location of Lesions](Final_Project/static/images/lesionLocation.png)
![Age Distribution](Final_Project/static/images/ageDistribution.png)
![Gender Distribution](Final_Project/static/images/gender.png)

Loading and resizing of images:

```skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))```

One of the challenges with our chosen data set was the imbalance of the available images. There were four times as many benign images as there were malignant images. We addressed this using an oversampling technique to augment our training image set.


### Step 3:  Building and Training the Model

We used the [Keras Sequential API](https://keras.io/getting-started/sequential-model-guide/) to build our convolutional neural network ("CNN"). The sequential model is a linear stack of layers. We chose to optimize the model using the Adam algorithm. This is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. We chose to address this imbalance by using an oversampling technique to reduce the imbalance.

* insert sample code + descriptions here

### Step 4: Testing the Model
* insert sample code + descriptions here
* insert model measurements here

## Conclusion
* Overall, we achieved xx% accuracy. Future work includes blah blah blah blah and blah.

## Deployment: Try it Yourself
A working prototype can be found here: [Heroku](https://lesionlegion.herokuapp.com)

## Project Team Members

* James Curtis
* Austen Manser
* Bill Nash
* Emanshu Patel
* Erica Unterreiner

## Acknowledgments

* Manu Siddhartha, Step Wise Approach: CCN Model [Kaggle](https://www.kaggle.com/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy)
* Marsh, Skin Lesion Analyzer + Tensorflow.js Web App [Kaggle](https://www.kaggle.com/vbookshelf/skin-lesion-analyzer-tensorflow-js-web-app)
* Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).

## Disclaimer 
The application and all code are provided as technical demonstration, not as medical product.
