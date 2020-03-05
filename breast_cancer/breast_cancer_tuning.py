import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

entrada = pd.read_csv('entrada_breast.csv')
classe = pd.read_csv('saida_breast.csv')

def createNet(optimizer, loss, kernel_initializer, activation, neurons):
    model = Sequential()
    #calc hidden layer (n. input + n. output) / 2 ==> (30+1) / 2 ==> 15.5
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    model.add(Dropout(0.2))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    
    return model
    
classificador = KerasClassifier(build_fn=createNet)

parameters = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tahn'],
              'neurons': [16, 10]}

grid_search = GridSearchCV(estimator=classificador, param_grid=parameters, scoring='accuracy', cv= 5)

grid_search = grid_search.fit(entrada, classe)

best_parameters = grid_search.best_estimator_
best_score = grid_search.best_score_