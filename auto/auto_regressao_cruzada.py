import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn

dataset = pd.read_csv('auto.csv', encoding='ISO-8859-1')

#drop column we won´t use
dataset = dataset.drop('dateCrawled', axis=1)
dataset = dataset.drop('dateCreated', axis=1)
dataset = dataset.drop('nrOfPictures', axis=1)
dataset = dataset.drop('postalCode', axis=1)
dataset = dataset.drop('lastSeen', axis=1)
dataset = dataset.drop('name', axis=1)
dataset = dataset.drop('seller', axis=1)
dataset = dataset.drop('offerType', axis=1)

#removing wrong prices
dataset = dataset[dataset.price > 10]
dataset = dataset[dataset.price < 350000]

dataset.loc[pd.isnull(dataset['vehicleType'])]
dataset['vehicleType'].value_counts()
dataset.loc[pd.isnull(dataset['gearbox'])]
dataset['gearbox'].value_counts()
dataset.loc[pd.isnull(dataset['model'])]
dataset['model'].value_counts()
dataset.loc[pd.isnull(dataset['fuelType'])]
dataset['fuelType'].value_counts()
dataset.loc[pd.isnull(dataset['notRepairedDamage'])]
dataset['notRepairedDamage'].value_counts()

#dict of values to fill
values_to_fill = {'vehicleType': 'limousine', 'gearbox': 'manuell',
                  'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}

dataset = dataset.fillna(value= values_to_fill)

previsores = dataset.iloc[: , 1:13].values
real_price = dataset.iloc[: , 0].values

#encoder
previsores[: ,0] = LabelEncoder().fit_transform(previsores[: ,0])
previsores[: ,1] = LabelEncoder().fit_transform(previsores[: ,1])
previsores[: ,3] = LabelEncoder().fit_transform(previsores[: ,3])
previsores[: ,5] = LabelEncoder().fit_transform(previsores[: ,5])
previsores[: ,8] = LabelEncoder().fit_transform(previsores[: ,8])
previsores[: ,9] = LabelEncoder().fit_transform(previsores[: ,9])
previsores[: ,10] = LabelEncoder().fit_transform(previsores[: ,10])

#onehotencoder
onehotencoder = OneHotEncoder(categorical_features= [0,1,3,5,8,9,10])

previsores = onehotencoder.fit_transform(previsores).toarray()

def createNet():
    regressor = Sequential()
    regressor.add(Dense(units= 160, activation= 'relu', input_dim= 323 ))
    regressor.add(Dense(units= 160, activation= 'relu'))
    regressor.add(Dense(units= 1, activation= 'linear'))

    regressor.compile(loss = 'mean_absolute_error', optimizer= 'adam', metrics= ['mean_absolute_error'])
    
    return regressor

regressor = KerasRegressor(build_fn= createNet, epochs= 100, batch_size= 300)

result = cross_val_score(estimator= regressor,
                         X= previsores, y= real_price,
                         cv= 10, scoring= 'neg_mean_absolute_error')

avg = result.mean()
desvio = result.std()