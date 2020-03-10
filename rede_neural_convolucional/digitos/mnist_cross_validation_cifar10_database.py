from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
(X, y), (X_test, y_test) = cifar10.load_data()

plt.imshow(X[0], cmap='gray')

previsores = X.reshape(X.shape[0], 32, 32, 3)
previsores = previsores.astype('float32')
previsores_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
previsores_test = previsores_test.astype('float32')

previsores /= 255
previsores_test /= 255

classe = np_utils.to_categorical(y, 10)
classe_test = np_utils.to_categorical(y_test, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())
classificador.add(Dense(units=256, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=256, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=10, activation='softmax'))

classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

classificador.fit(previsores, classe, batch_size=128, epochs=5)

precisao = classificador.evaluate(previsores_test, classe_test)