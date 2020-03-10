from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

(X, y), (X_test, y_test) = mnist.load_data()

previsores = X.reshape(X.shape[0], 28, 28, 1)
previsores_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
previsores = previsores.astype('float32')
previsores_test = previsores.astype('float32')

previsores /= 255
previsores_test /= 255

classe_treinamento = np_utils.to_categorical(y, 10)
classe_teste = np_utils.to_categorical(y, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2,2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=10, activation='softmax'))

classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

#creates images from training base. (rotate, change pixel, etc)
gerador_treinamento = ImageDataGenerator(rotation_range=7,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         height_shift_range=0.07,
                                         zoom_range=0.2)

gerador_test = ImageDataGenerator()

train_data = gerador_treinamento.flow(previsores, classe_treinamento, batch_size=128)
test_data = gerador_test.flow(previsores_test, classe_teste, batch_size=128)

classificador.fit_generator(train_data, steps_per_epoch=60000 / 128, epochs=5,
                            validation_data=test_data, validation_steps=10000/128)