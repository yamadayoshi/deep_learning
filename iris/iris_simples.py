import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#lib to transform string to "id"
from sklearn.preprocessing import LabelEncoder
#lib to transform value´s shape
from keras.utils import np_utils

dataset = pd.read_csv('iris.csv')

entrada = dataset.iloc[:,0:4].values
classe = dataset.iloc[:,4].values

#transform 'iris-x' to id
classe = LabelEncoder().fit_transform(classe)

classe_dummy = np_utils.to_categorical(classe)

#iris setosa ==> output 1 0 0 
#iris versicolor ==> output 0 1 0
#iris virginica ==> output 0 0 1

previsores_treinamento, previsores_test, classe_treinamento, classe_test = train_test_split(entrada, classe_dummy, test_size=0.25)

model = Sequential()
model.add(Dense(units=8, activation='tanh', kernel_initializer='normal', input_dim=4))
model.add(Dropout(0.2))
model.add(Dense(units=8, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(previsores_treinamento, classe_treinamento, epochs=100, batch_size=10)

model_json = model.to_json()
with open('iris.json', 'w') as json_file:
    json_file.write(model_json)
    
model.save_weights('iris.h5')

#check the loss and accuracy
result = model.evaluate(previsores_test, classe_test)

#check all the 3 outputs
previsao = model.predict(previsores_test)

#create list with the higher val
classe_test_norm = [np.argmax(t) for t in classe_test]

#create list with the higher val
previsao_norm = [np.argmax(t) for t in previsao]

matrix = confusion_matrix(previsao_norm, classe_test_norm)