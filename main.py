import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import pandas as pd
from sklearn import preprocessing
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation
from keras.callbacks import  TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

sequence_length = 60
data_dim = 9
predict_length = 3
epoch = 10
batch_size = 64


def classify(current, future):
    return future - current


def preprocess(df):

    for col in df.columns:
        if col != 'Target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
    # print(df['Weighted_Price'].shape)

    seq_data = []
    prev_days = deque(maxlen=sequence_length)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == sequence_length:
            seq_data.append([np.array(prev_days), i[-1]])

    random.shuffle(seq_data)

    x = []
    y = []
    # print(len(seq_data))

    for seq, target in seq_data:
        x.append(seq)
        y.append(abs(target))

    # print(len(x), len(y))
    return np.array(x), y


data = pd.read_csv('D:\\Data\\bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv', engine='python')  # get data
data = data[data.Weighted_Price >= 0]  # remove nan
data = data[(data.index >= 3003136)]
data['Future'] = data['Weighted_Price'].shift(-predict_length)
data['Target'] = list(map(classify, data['Close'], data['Future']))
data.index = range(len(data))  # re-index the dataframe

last_five = data.index.values[-int(0.05*len(data))]  # get a time index for the last 5 percent of data
validation = data[(data.index >= last_five)]  # create validation as last 5 percent of data
data = data[(data.index <= last_five)]  # remove validation from the normal dataframe

x_train, y_train = preprocess(data)
x_val, y_val = preprocess(validation)

model = Sequential()
'''
model.add(LSTM(32, return_sequences=True,
               input_shape=(SEQUENCE_LENGTH, data_dim), activation='relu'))

model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(LSTM(32, return_sequences=True, activation='tanh'))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(32, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add((Dense(3, activation='softmax')))
'''
model.add(LSTM(32, return_sequences=True,
              input_shape=(sequence_length, data_dim), activation='relu'))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(32, return_sequences=True, activation='relu'))  # returns a sequence of vectors of dimension 32
model.add(Dropout(.2))
model.add(BatchNormalization())

model.add(LSTM(32, activation='tanh'))  # return a single vector of dimension 32
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(1000, activation='relu'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.01, decay=1e-6), metrics=['accuracy'])

print(model.input_shape)
print(x_train.shape)
model.fit(x_train, y_train, batch_size=32, epochs=epoch, validation_data=(x_val, y_val))


