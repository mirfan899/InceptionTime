import pandas as pd
import numpy as np
from keras.preprocessing import sequence

WAVE_TYPE = {"down": 0, "up": 1}
INVERSE_WAVE_TYPE = {0: "down", 1: "up"}
data = pd.read_csv("data/5min-patterns.csv")

total_features = []
max_count = []
ups = []
downs = []

data["wave_type"] = data["wave_type"].map({"down": 0, "up": 1})
data['value_grp'] = (data["wave_type"].diff(1) != 0).astype('int').cumsum()
total_groups = data["value_grp"].max()
# for grp in range(len(total_groups)):
for grp in range(total_groups):
    features = []
    rows = data[data["wave_grp_id"] == grp]
    max_count.append(len(rows.index))

    for i in rows.index:
        features.append(rows["cp"][i])
        features.append(rows["kvFst"][i])
        features.append(rows["kvTrgger"][i])
        features.append(rows[" ripple_angle"][i])
        features.append(rows["wave_angle"][i])
    features.insert(0, rows["wave_type"][rows.index[0]])
    total_features.append(features)
total_features = sequence.pad_sequences(total_features, maxlen=101, padding='post', dtype='float', truncating='post')
print(np.array(total_features).shape)
print(max(max_count))
print(min(max_count))
with open("features/Wave_Type.csv", "w") as f:
    for line in total_features:
        feats = [str(i) for i in line]
        f.write(",".join(feats) + "\n")
