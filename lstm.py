from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from sklearn.model_selection import train_test_split

WAVE_TYPE = {"down": 0, "up": 1}

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = read_csv("data/5min-patterns.csv")
data.drop(columns=['ripple_id', 'ripple_type', 'wave_grp_id', 'date_and_time'], inplace=True)

data["wave_type"] = data["wave_type"].map(WAVE_TYPE)
y = data["wave_type"].astype(int).values
data.drop(columns=["wave_type"], inplace=True)

X = data.values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test= x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape, y_train.shape, x_test.shape, y_train.shape)

# design network
model = Sequential()
model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
# fit network
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=1,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
# yhat = model.predict(x_test)
