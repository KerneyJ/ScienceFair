import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
from keras.models import load_model

sequence_length = 60
data_dim = 9
predict_length = 3
epoch = 5
batch_size = 500


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess(df):
    for col in df.columns:
        if col != 'Target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    seq_data = []
    prev_days = deque(maxlen=sequence_length)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == sequence_length:
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
data = data[(data.index >= 3403136)]
data['Future'] = data['Weighted_Price'].shift(-predict_length)
data['Target'] = list(map(classify, data['Close'], data['Future']))
data.index = range(len(data))  # re-index the dataframe

l20 = data.index.values[-int(0.20*len(data))]  # get a time index for the last 5 percent of data
validation = data[(data.index >= l20)]  # create validation as last 5 percent of data
data = data[(data.index <= l20)]  # remove validation from the normal dataframe

x_train, y_train = preprocess(data)
x_val, y_val = preprocess(validation)

model = load_model('science_fair.h5')

print(x_val[0].reshape((1, 60, 9)).shape)

for i in range(len(x_val)):
    print(model.predict( x_val[0].reshape((1, 60, 9)) ), y_val[i])
