# Facial Emotion Recognition (FER) Predictor

<img src="./media/banner.png" alt="drawing" width="1000"/>

## Table of Contents
- [Project Overview](#project-overview)
- [Getting started immediately](#getting-started)
- [Datasets](#datasets)
- [Dependencies](#dependencies)
- [Workflow](#workflow)
- [Directory structure](#dirctory-structure)
- [Models](#models)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contributions and Feedback](#contributions-and-feedback)

---

## [Project Overview](#project-overview)

Understanding human emotions is crucial for human-computer interaction, artificial intelligence, and affective computing. This project delves into the realm of facial emotion recognition:
* using the Kaggle Facial [Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset
* building a [Convolutional Neural Network (CNN) Tensorflow/Keras model](./models/emotion_classifier.h5), and
* deploying it on AWS which can be accessed through a REST-API.

As emotions play a pivotal role in communication, this project aims to contribute to the evolving landscape of emotion-aware technology. Whether applied in virtual assistants, sentiment analysis, or interactive systems, accurate facial emotion recognition enhances user experience and engagement. Dive into the repository to explore the code, contribute to advancements, and potentially integrate the model into your projects for a more emotionally intelligent interface.

## [Datasets](#datasets)

The FER2013 dataset, curated by Pierre-Luc Carrier and Aaron Courville, was developed for the Kaggle competition titled [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). All the dataset files can be downloaded and decompressed from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). This dataset provides a valuable resource for exploring challenges in representation learning and advancing facial expression recognition algorithms. There are more details about the dataset [**here**](./data/README.md).

## [Workflow](#workflow)

### 1. Cloning the repository: 

```bash
git clone git@github.com:abhirup-ghosh/facial-expression-classifier-app.git
```

### 2. **Setting up the environment:**

The easiest way to set up the environment is to use [Anaconda](https://www.anaconda.com/download). I used the standard Machine Learning Zoomcamp conda environment `ml-zoomcamp`, which you can create, activate, and install the relevant libraries in, using the following commands in your terminal:

```bash
conda create -n ml-zoomcamp python=3.9 -y
conda activate ml-zoomcamp
conda install numpy pandas scikit-learn seaborn jupyter tensorflow -y

# install tflite-runtime
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

Alternatively, I have also provided a conda `environment.yml` file that can be directly used to create the environment:

```bash
conda env create -f opt/environment.yml
```

### 3. Running `notebooks/notebook.ipynb`

You can run the notebook on **[Google Colab](https://colab.research.google.com/github/abhirup-ghosh/facial-expression-classifier-app/blob/main/notebooks/notebook.ipynb)**. Details of the instance are:
* Notebook: Python3 
* Hardware: Single T4 GPU

```
Package     : Verion
--------------------
pandas      : 1.5.3
numpy       : 1.23.5
matplotlib  : 3.7.1
seaborn     : 0.12.2
sklearn     : 1.2.2
pickle      : 4.0
tensorflow  : 2.14.0
```



This notebook outlines the entire investigation and consists of the following steps [🚨 Skip this step, if you want to directly want to use the final configuration for training and/or final model for predictions]:

- Data Loading: `data/icml_face_data.csv`  
- Data preprocessing
- Exploratory data analysis
- Setting up a validation framwork
- Model definition: CNN (+ data augmentation)
- Model evaluation [and hyper-parameter tuning]
- Saving the best model: `models/emotion_classifier.h5`
- Preparation of the test data
- Making predictions using the saved model
- Convert to TF-Lite model: `models/emotion_classifier.tflite`
- Remove TF dependency

### 4. **Training model**
We encode our best, tuned CNN model inside the scripts/train.py file which can be run using:

```bash
cd scripts
python train.py
```
The output of this script can be found in: `models/*`. It has an average accuracy of 65% and will be used for the following steps. The training script also converts the keras model into a light-weight TF-lite model.

### 5. **Prepare Code for Lambda**

#### 5.1. Running `notebooks/notebook-lambda.ipynb`

This notebook provides a step-by-step guide for preparing the Lambda Code:
* Use `tflite_runtime` model instead of Keras model
* Remove any tensorflow/keras dependency on preprocessing/prediction
* Define `predict(url)` function to make prediction on an **image url (input)**; output is a dictionary of emotion classes with respective probabilities.
* Create  `lambda_handler(event, context)` wrapper for calling `predict(url)`

#### 5.2. `scripts/lambda_function.py`

Once we test out the entire lambda framework using the above notebook, we convert it into a python script.

### 6. Make predictions

**Test Image: Happy**
<img src="./media/test_image_happy.jpeg" alt="drawing" width="1000"/>

```bash
facial-expression-classifier-app/scripts> python

Python 3.9.18 (main, Sep 11 2023, 08:20:50) 
[Clang 14.0.6 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.

>>> import lambda_function

>>> event = {'url': 'https://upload.wikimedia.org/wikipedia/commons/0/09/The_joy_of_the_happy_face_by_Rasheedhrasheed.jpg'}

>>> lambda_function.lambda_handler(event, None)

{'Anger': 8.256378446588042e-35, 'Disgust': 2.9382650407717056e-39, 'Fear': 2.9382650407717056e-39, 'Happy': 1.0, 'Neutral': 0.0, 'Sadness': 0.0, 'Surprise': 0.0}
```

**Prediction: Happy**


### 6. Deployment on Lambda



## Data Citation

[1] "Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013.