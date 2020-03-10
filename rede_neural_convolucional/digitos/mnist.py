import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#load mnist data
(X_training, y_training), (X_test, y_test) = mnist.load_data()

#show image and convert img to gray
plt.title('Class ' + str(y_training[0]))
plt.imshow(X_training[0], cmap= 'gray')

#28 height, 28 width, 1 channel (rgb)
previsores_training = X_training.reshape(X_training.shape[0], 28, 28, 1)
previsores_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#convert to float32 to get 0.00001 to 1
previsores_training = previsores_training.astype('float32')
previsores_test = previsores_test.astype('float32')

#normalize data to gain performace
previsores_training /= 255
previsores_test /= 255

#normalize output 
training_class = np_utils.to_categorical(y_training, 10)
test_class = np_utils.to_categorical(y_test, 10)

classificador = Sequential()
#first convolution layer
classificador.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
#normalize features
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

#second convolution layer
classificador.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
#normalize features
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

#Neural Net
classificador.add(Dense(units= 128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units= 128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units= 10, activation='softmax'))

classificador.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

classificador.fit(previsores_training, training_class, batch_size=125, epochs=5)

result = classificador.evaluate(previsores_test, test_class)