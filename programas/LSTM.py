import numpy as np 
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import pandas as pd 

df = pd.read_csv('../dataset/casos/auto-clean.csv')

fstring = ''
for row in df['descripcion'][df['responsabilidad']!='COMPROMETIDA']:
    fstring+=row.strip()+' '

proccessed_inputs = fstring.split()
chars = sorted(list(set(proccessed_inputs)))
input_len = len(proccessed_inputs)
vocab_len = len(chars)
print(input_len,vocab_len)

chars2num = dict((c,i) for i,c in enumerate(chars))


seq_len = 100
x_data, y_data = [],[]

for i in range(input_len-seq_len):
    in_seq = proccessed_inputs[i:i+seq_len]
    out_seq = proccessed_inputs[i+seq_len]
    x_data.append([chars2num[char] for char in in_seq])
    y_data.append([chars2num[out_seq]])

n_patterns = len(x_data)
print(n_patterns)

x = np.reshape(x_data,(n_patterns,seq_len,1 ))
x = x/float(vocab_len)
y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256,input_shape = (x.shape[1],x.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam')

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

model.fit(x, y, epochs=10, batch_size=256, callbacks=desired_callbacks)

filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i,c) for i,c in enumerate(chars))
start = np.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ' '.join([num_to_char[value] for value in pattern]), "\"")

for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = num_to_char[index]

    sys.stdout.write(result+' ')

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
