import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from keras.models import model_from_json

data = pd.read_csv('ODI_training.csv')
data = data.drop(['team1','Date','team2','Ground'], axis =1)

#data = shuffle(data)

#for test data------------------------------------------
data_test = pd.read_csv('ODI_predict_data.csv')
data_test = data_test.drop(['team1','Date','team2','Ground'], axis =1)

#data_test = shuffle(data_test)

Y = data['result'] # final name of winning team, fetch from training data

# Transform name species into numerical values 
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)
#print(Y)

#for test data-----------------------------------------------------
X_test = data_test.drop(['result'], axis = 1)
#X_test = data.dropna(axis = 0, how ='any') 

scaler = MinMaxScaler(feature_range=(0, 1))
X_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)

X_test = np.array(X_test)
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
for i in range(0,len(data_test)):

	pred = loaded_model.predict_proba(X_test)

		#print("top 10 best matches brands percentage")

	dfe = np.sort(pred[i]*100)[:-3:-1] # sorting in percentage outcome
	wer = np.argsort(pred[i]*100)[:-3:-1] # sorting in brand_id outcome
	wer = encoder.inverse_transform(wer)

	#print(pred)
	#print("_______________________________________________________________")
	#print(dfe)

	abc = dict(zip(wer,dfe))
	print(abc)
	
