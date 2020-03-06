import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('iris.csv')

entrada = dataset.iloc[:, 0:4].values
classe = dataset.iloc[:, 4].values

#normaliza
#transform label in id
classe = LabelEncoder().fit_transform(classe)

#create matrix value from classe
class_dummy = np_utils.to_categorical(classe)

def createNet():
    model = Sequential()
    model.add(Dense(units=3, activation='relu', kernel_initializer='random_uniform', input_dim=4))
    model.add(Dropout(0.1))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

classifier = KerasClassifier(build_fn= createNet, epochs=100, batch_size= 10)

result = cross_val_score(estimator= classifier, X= entrada, y= classe, cv= 10, scoring='accuracy')

#calculate avg
avg = result.mean()

#desvio
desvio = result.std()