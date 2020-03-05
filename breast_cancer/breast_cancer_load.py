import numpy as np
from keras.models import model_from_json

#read json file
file = open('breast_model.json', 'r')
network = file.read()
file.close()

#load model from json and weights
model = model_from_json(network)
model.load_weights('breast_weights.h5')

novo = np.array([[10.2,5.6,155.0,15.4,18.5,75.5,15.9,79.4,56.9,15, 10.2,5.6,155.0,15.4,18.5,75.5,15.9,79.4,56.9,15, 10.2,5.6,155.0,15.4,18.5,75.5,15.9,79.4,56.9,15]])

previsao = model.predict(novo) > 0.8