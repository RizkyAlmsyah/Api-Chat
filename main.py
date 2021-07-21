#!/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

factory = StemmerFactory()
stemmer = factory.create_stemmer()

factory_remover = StopWordRemoverFactory()
stopword = factory_remover.create_stop_word_remover()

def preprocessing_indo(message):
    pesan_bagus = []
    for i in message:
        stop = stopword.remove(i.lower())
        output = stemmer.stem(stop)
        pesan_bagus.append(output)
        return pesan_bagus


@app.route('/predict-mental-health/<message>', methods=['GET'])
def getPreditMental(message):
    tokenizer = Tokenizer(num_words=5000, oov_token='x')
    df = pd.read_excel('data/dataset.xlsx')
    label = pd.get_dummies(df.V1)
    df_label = pd.concat([df, label], axis=1)
    df_label = df_label.drop(columns='V1')
    text = df_label['V2'].astype(str)
    train = preprocessing_indo(text)
    
    tokenizer.fit_on_texts(train)

    stop = stopword.remove(message.lower())
    output = stemmer.stem(stop)
    
    sequence = tokenizer.texts_to_sequences([output])
    # pad the sequence
    sequence = pad_sequences(sequence)
    # get the prediction
    reconstructed_model = keras.models.load_model("data/model.h5")

    prediction =  reconstructed_model.predict(sequence)[0]
    mental_health = np.argmax(prediction)
    if mental_health == 1:
        label = "Depresi"
    else:
        label = "Skizopenia"
    return jsonify({'prediction': label,
                    'stemmer': output.split()})

app.run()