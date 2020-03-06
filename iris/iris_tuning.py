import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('iris.csv')

entrada = dataset.iloc[:,0:4].values
classe = dataset.iloc[:,4].values



def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_dim = 4))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loos': ['categorical_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5)
grid_search = grid_search.fit(entrada, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_