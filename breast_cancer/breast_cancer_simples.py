import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense

entrada = pd.read_csv('entrada_breast.csv')
classe = pd.read_csv('saida_breast.csv')

#split data
entrada_treinamento, entrada_teste, classe_treinamento, classe_teste = train_test_split(entrada, classe, test_size=0.25)

model = Sequential()

#calc hidden layer (n. input + n. output) / 2 ==> (30+1) / 2 ==> 15.5
model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
model.add(Dense(units=1, activation='sigmoid'))

otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)

#using otimizador
#model.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

#binary_crossentropy = for problem binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(entrada_treinamento, classe_treinamento, epochs=100, batch_size=10)

#get weights from layers
peso0 = model.layers[0].get_weights()
peso1 = model.layers[1].get_weights()
peso2 = model.layers[2].get_weights()

previsores = model.predict(entrada_teste)
#convert previsores to bool
previsores = (previsores > 0.5)
precisao = accuracy_score(classe_teste, previsores)
matrix = confusion_matrix(classe_teste, previsores)

resultado = model.evaluate(entrada_teste, classe_teste)