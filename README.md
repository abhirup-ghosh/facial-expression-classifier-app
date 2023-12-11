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

## [Datasets](#datasets)

The FER2013 dataset, curated by Pierre-Luc Carrier and Aaron Courville, was developed for the Kaggle competition titled [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). All the dataset files can be downloaded and decompressed from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). This dataset provides a valuable resource for exploring challenges in representation learning and advancing facial expression recognition algorithms. There are more details about the dataset [**here**](./data/README.md).

## [Workflow](#workflow)

### 1. Cloning the repository: 

```
git clone git@github.com:abhirup-ghosh/facial-expression-classifier-app.git
```

### 2. **Setting up the environment:**

The easiest way to set up the environment is to use [Anaconda](https://www.anaconda.com/download). I used the standard Machine Learning Zoomcamp conda environment `ml-zoomcamp`, which you can create, activate, and install the relevant libraries in, using the following commands in your terminal:

```
conda create -n ml-zoomcamp python=3.9
conda activate ml-zoomcamp
conda install numpy pandas scikit-learn seaborn jupyter tensorflow
```

Alternatively, I have also provided a conda `environment.yml` file that can be directly used to create the environment:

```
conda env create -f opt/environment.yml
```

### 3. Running `notebooks/notebook.ipynb`


## Data Citation

[1] "Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013.