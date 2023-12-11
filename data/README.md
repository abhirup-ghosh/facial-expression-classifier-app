# Data: FER2013 (Facial Expression Recognition 2013 Dataset)

This dataset was prepared by  Pierre-Luc Carrier and Aaron Courville for the Kaggle competition:
[Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). All the files can be downloaded and decompressed from:

```
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
```




## About Dataset

The dataset comprises 48x48 pixel grayscale images depicting faces. The faces have been automatically aligned, ensuring centrality and uniform spatial occupancy in each image. The objective is to classify each face into one of seven emotion categories based on the displayed facial expression. These emotions/categories are:

| Emotion Code | Emotion      |
|--------------|--------------|
| 0            | Angry        |
| 1            | Disgust      |
| 2            | Fear         |
| 3            | Happy        |
| 4            | Sad          |
| 5            | Surprise     |
| 6            | Neutral      |

![alt text](../media/emotions.png)

## Files

**Relevant files:**
* **train.csv:** training data -- contains two columns: "emotion" and "pixels." The "emotion" column contains a numeric code (0 to 6) representing the emotion in the image, while the "pixels" column contains a string enclosed in quotes for each image. This string comprises space-separated pixel values arranged in row-major order; contains 28,709 examples.
* **test.csv:** test data -- includes only the "pixels" column, and the objective is to predict the corresponding emotion.
* **icml_face_data.csv/fer2013.csv** -- contains all data (training + test) in three columns: "emotion", "pixels" and "Usage" (whether to be used for "Training", "PublicTest" and "PrivateTest")
* **fer2013.bib:** citation for dataset


**All Files:**
```bash
./data/challenges-in-representation-learning-facial-expression-recognition-challenge/
├── example_submission.csv
├── fer2013
│   ├── README
│   ├── fer2013.bib
│   └── fer2013.csv
├── icml_face_data.csv
├── test.csv
└── train.csv
```

## Citation

"Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013.

```
@MISC{Goodfeli-et-al-2013,
       author = {Goodfellow, Ian and Erhan, Dumitru and Carrier, Pierre-Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and Zhou, Yingbo and Ramaiah, Chetan and Feng, Fangxiang and Li, Ruifan and Wang, Xiaojie and Athanasakis, Dimitris and Shawe-Taylor, John and Milakov, Maxim and Park, John and Ionescu, Radu and Popescu, Marius and Grozea, Cristian and Bergstra, James and Xie, Jingjing and Romaszko, Lukasz and Xu, Bing and Chuang, Zhang and Bengio, Yoshua},
     keywords = {competition, dataset, representation learning},
        title = {Challenges in Representation Learning: A report on three machine learning contests},
         year = {2013},
  institution = {Unicer},
          url = {http://arxiv.org/abs/1307.0414},
     abstract = {The ICML 2013 Workshop on Challenges in Representation
Learning focused on three challenges: the black box learning challenge,
the facial expression recognition challenge, and the multimodal learn-
ing challenge. We describe the datasets created for these challenges and
summarize the results of the competitions. We provide suggestions for or-
ganizers of future challenges and some comments on what kind of knowl-
edge can be gained from machine learning competitions.

http://deeplearning.net/icml2013-workshop-competition}
}
```

