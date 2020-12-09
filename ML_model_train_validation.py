import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy as sp 
import sklearn
import random 
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, model_selection
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from keras.models import model_from_json
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization


data = pd.read_csv('ODI_trainig.csv')
data = data.drop(['team1','Date','team2','Ground'], axis =1)

#data = shuffle(data)

data = data.reset_index(drop = True)

X = data.drop(['result'], axis = 1)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

X = np.array(X)
Y = data['result']

# Transform name species into numerical values 
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)


input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(20,input_dim = input_dim , activation = 'relu'))
#model.add(Dense(135, activation = 'relu'))
#model.add(Dense(100, activation = 'relu'))
#model.add(BatchNormalization())
#model.add(Dense(600))
#model.add(LeakyReLU(alpha=[0.05]))
model.add(BatchNormalization())
model.add(Dense(8))
model.add(LeakyReLU(alpha=[0.05]))
model.add(BatchNormalization())
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(X, Y, epochs = 10, batch_size = 10)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
