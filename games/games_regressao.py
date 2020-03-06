import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('games.csv')

dataset = dataset.drop('Other_Sales', axis= 1)
dataset = dataset.drop('Global_Sales', axis= 1)
dataset = dataset.drop('Developer', axis= 1)

dataset = dataset.dropna(axis= 0)

dataset = dataset.loc[dataset['NA_Sales'] > 1]
dataset = dataset.loc[dataset['EU_Sales'] > 1]

game_names = dataset.Name
dataset = dataset.drop('Name', axis= 1)

previsores = dataset.iloc[:, [0,1,2,3,7,8,9,10,11]].values
sales_na = dataset.iloc[:, 4].values
sales_eu = dataset.iloc[:, 5].values
sales_jp = dataset.iloc[:, 6].values

previsores[:, 0] = LabelEncoder().fit_transform(previsores[:, 0])
previsores[:, 2] = LabelEncoder().fit_transform(previsores[:, 2])
previsores[:, 3] = LabelEncoder().fit_transform(previsores[:, 3])
previsores[:, 8] = LabelEncoder().fit_transform(previsores[:, 8])

onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])

previsores = onehotencoder.fit_transform(previsores).toarray()

def customActivation(x):
    return np.ext(x) * -1 

activation = Activation(customActivation, name='activelife')

#creating layers manually
camada_entrada = Input(shape=(61,))
camada_oculta1 = Dense(units= 32, activation= 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units= 32, activation= 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units= 1, activation= 'linear')(camada_oculta2)
camada_saida2 = Dense(units= 1, activation= 'linear')(camada_saida1)
camada_saida3 = Dense(units= 1, activation= 'linear')(camada_saida2)

regressor = Model(inputs= camada_entrada, outputs= [camada_saida1, camada_saida2, camada_saida3])

regressor.compile(optimizer= 'adam', loss= 'mse')

regressor.fit(previsores, [sales_na, sales_eu, sales_jp])

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)