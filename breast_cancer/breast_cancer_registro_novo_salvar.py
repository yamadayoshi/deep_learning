import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

entrada = pd.read_csv('entrada_breast.csv')
classe = pd.read_csv('saida_breast.csv')

model = Sequential()
#calc hidden layer (n. input + n. output) / 2 ==> (30+1) / 2 ==> 15.5
model.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
model.add(Dropout(0.2))
model.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(entrada, classe, batch_size=10, epochs=100)

#save model
model_json = model.to_json()
with open('breast_model.json', 'w') as json_file:
    json_file.write(model_json)

#save weights
model.save_weights('breast_weights.h5')

novo = np.array([[10.2,5.6,155.0,15.4,18.5,75.5,15.9,79.4,56.9,15, 10.2,5.6,155.0,15.4,18.5,75.5,15.9,79.4,56.9,15, 10.2,5.6,155.0,15.4,18.5,75.5,15.9,79.4,56.9,15]])

previsao = model.predict(novo) > 0.8