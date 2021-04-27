from flask import Flask
from flask_restful import Resource, Api, reqparse
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
from tensorflow import keras
import numpy as np
INVERSE_WAVE_TYPE = {0: "down", 1: "up"}

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model("models/best_model.h5")


def get_wave_type(row):
    if row:
        signal = row.split(",")
        signal = np.array([float(x) for x in signal])
        features = sequence.pad_sequences([signal], maxlen=101, padding='post', dtype='float', truncating='post')
        pred = model.predict(features, batch_size=32)
        wave_type = np.argmax(pred, axis=1)
        return {"wave_type": INVERSE_WAVE_TYPE[wave_type[0]]}
    else:
        return {"message": "provide the row data."}


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("row", required=True, type=str, help="please provide the row to get wave_type")


class WaveClass(Resource):
    def get(self):
        return {"message": "Welcome to InceptionTime API", "status": 200}

    def post(self):
        args = parser.parse_args()
        if args["row"]:
            result = get_wave_type(args["row"])
            return result
        else:
            return {"You should provide a sentence."}


api.add_resource(WaveClass, '/api/wave_type')

if __name__ == '__main__':
    app.run(debug=False, threaded=False, processes=1)
