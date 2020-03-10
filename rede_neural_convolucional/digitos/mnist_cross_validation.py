from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(X, y), (X_test, y_test) = mnist.load_data()

previsores = X.reshape(X.shape[0], 28, 28, 1)
previsores = previsores.astype('float32')

previsores /= 255

classe = np_utils.to_categorical(y, 10)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

results = []

for train_index, test_index in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
    classificador = Sequential()
    classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dense(units=10, activation='softmax'))
    
    classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
    
    classificador.fit(previsores[train_index], classe[train_index], batch_size=128, epochs=5)
    
    precisao = classificador.evaluate(previsores[test_index], classe[test_index])
    
    results.append(precisao[1])
    
mean = sum(results) / len(results)
