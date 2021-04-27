from tensorflow import keras
import numpy as np
import math
from keras.preprocessing import sequence
model = keras.models.load_model("models/best_model.h5")
signal = "1109.25,0.0,0.0,0.0,1111.0,-1.81898940354586e-12".split(",")

signal = np.array([math.ceil(float(x)) for x in signal])
signal = sequence.pad_sequences([signal], maxlen=101, padding='post', dtype='float', truncating='post')

print(signal.shape)
pred = model.predict(signal, batch_size=64)
y_pred = np.argmax(pred, axis=1)
print(y_pred)
print(y_pred[0])