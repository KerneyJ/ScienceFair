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

SEQUENCE_LENGTH = 60
data_dim = 9
PREDICT_LENGTH = 3
epoch = 1
BATCH_SIZE = 64


def classify(current, future):
    return future - current

def preprocess(df):
    for col in df.columns:
        if col != 'Target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    seq_data = []
    prev_days = deque(maxlen=SEQUENCE_LENGTH)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQUENCE_LENGTH:
            seq_data.append([np.array(prev_days), i[-1]])

    random.shuffle(seq_data)

    # Balance data
    buy = []
    sell = []

    for seq, target in seq_data:
        if target == 0:
            sell.append([seq, target])
        elif target == 1:
            buy.append([seq, target])

    random.shuffle(sell)
    random.shuffle(buy)

    low = min(len(sell), len(buy))

    buy = buy[:low]
    sell = sell[:low]
    seq_data = buy+sell
    random.shuffle(seq_data)

    x = []
    y = []

    for seq, target in seq_data:
        x.append(seq)
        y.append(target)

    return np.array(x), y


data = pd.read_csv('D:\\Data\\bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv', engine='python')  # get data
data = data[data.Weighted_Price >= 0]  # remove nan
data = data[(data.index >= 3203136)]
data['Future'] = data['Weighted_Price'].shift(-PREDICT_LENGTH)
data['Target'] = list(map(classify, data['Close'], data['Future']))
data.index = range(len(data))  # re-index the dataframe

last_five = data.index.values[-int(0.05*len(data))]  # get a time index for the last 5 percent of data
validation = data[(data.index >= last_five)]  # create validation as last 5 percent of data
data = data[(data.index <= last_five)]  # remove validation from the normal dataframe

x_train, y_train = preprocess(data)
x_val, y_val = preprocess(validation)

model = Sequential()
#model.add(LSTM(128, input_shape=(60, 9), return_sequences=True, activation='tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

#model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True, activation='tanh'))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())

#model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True, activation='tanh'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))

#model.add((Dense(2, activation='softmax')))

model.add(LSTM(32, return_sequences=True,
               input_shape=(SEQUENCE_LENGTH, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.01, decay=1e-6), metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=500, epochs=epoch, validation_data=(x_val, y_val))
print(x_train[0].shape)
