# from tensorflow import device to use gpu with device('/device:GPU:0'):
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras.models import Sequential
from keras.layers import LSTM, Dense

import numpy as np

data_dim = 1
timesteps = 1
num_classes = 1
batch_size = 1

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((2388829, timesteps, data_dim))
y_train = np.random.random((2388829, num_classes))
print(x_train.shape, y_train.shape)
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=1, shuffle=False,)
model.save('big_model.h5')
testX = np.random.random((1, 60, 1))
print(testX.shape)
print(model.predict(testX).shape)