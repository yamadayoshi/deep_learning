import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

entrada = pd.read_csv('entrada_breast.csv')
classe = pd.read_csv('saida_breast.csv')

def createNet():
    model = Sequential()
    #calc hidden layer (n. input + n. output) / 2 ==> (30+1) / 2 ==> 15.5
    model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    return model

classificador = KerasClassifier(build_fn= createNet, epochs=100, batch_size=10)

resultados = cross_val_score(estimator=classificador, X=entrada, y=classe, cv=10, scoring='accuracy')

media = resultados.mean()

desvio = resultados.std()