#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np


interpreter = tflite.Interpreter(model_path='../models/emotion_classifier.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]


def prepare_input(url):
    
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    img = img.convert('L')
    img = img.resize((48, 48), Image.NEAREST)
    x = np.array(img)
    X = np.array([x])
    X = np.expand_dims(np.float32(X), axis=-1)

    return X

def predict(url):
    X = prepare_input(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(class_labels, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
