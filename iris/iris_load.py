from keras.models import model_from_json
import numpy as np

file = open('iris.json', 'r')
network = file.read()
file.close()

model = model_from_json(network)

model.load_weights('iris.h5')

novo = np.array([[0.8,0.1,0.5,0.1]])

predict = [np.argmax(t) for t in model.predict(novo)]