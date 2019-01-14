import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
# (356662, 60, 9)
data_dim = 9
timesteps = 60
output_shape = 1

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(output_shape, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((356662, timesteps, data_dim))
y_train = np.random.random((356662, output_shape))

print(model.input_shape)
model.fit(x_train, y_train,
          batch_size=500, epochs=1)

print(model.predict(np.random.random((5, 60, 9))))
