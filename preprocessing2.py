import pandas as pd
import numpy as np
from keras.preprocessing import sequence

WAVE_TYPE = {"down": 0, "up": 1}
# data = pd.read_csv("180min-patterns.csv")
data = pd.read_csv("data/5min-patterns.csv")
total_groups = data["wave_grp_id"].max()

total_features = []
max_count = []
# for grp in range(len(total_groups)):
for grp in range(total_groups):
    features = []
    rows = data[data["wave_grp_id"] == grp]
    max_count.append(len(rows.index))

    for i in rows.index[:50]:
        features.append(rows["cp"][i])
        features.append(rows["kvFst"][i])
        features.append(rows["kvTrgger"][i])
        # features.append(rows["ripple_angle"][i])
        features.append(rows["wave_angle"][i])
    features.insert(0, WAVE_TYPE[rows["wave_type"][rows.index[0]]])
    total_features.append(features)

total_features = sequence.pad_sequences(total_features, maxlen=200, padding='post', dtype='float', truncating='post')
print(np.array(total_features).shape)

with open("Wave_TRAIN.csv", "w") as f:
    for line in total_features:
        feats = [str(i) for i in line]
        f.write(",".join(feats) + "\n")


data = pd.read_csv("data/180min-patterns.csv")
total_groups = data["wave_grp_id"].max()

total_features = []
max_count = []
# for grp in range(len(total_groups)):
for grp in range(total_groups):
    features = []
    rows = data[data["wave_grp_id"] == grp]
    max_count.append(len(rows.index))

    for i in rows.index[:50]:
        features.append(rows["cp"][i])
        features.append(rows["kvFst"][i])
        features.append(rows["kvTrgger"][i])
        # features.append(rows["ripple_angle"][i])
        features.append(rows["wave_angle"][i])
    features.insert(0, WAVE_TYPE[rows["wave_type"][rows.index[0]]])
    total_features.append(features)

total_features = sequence.pad_sequences(total_features, maxlen=200, padding='post', dtype='float', truncating='post')
print(np.array(total_features).shape)

with open("Wave_TEST.csv", "w") as f:
    for line in total_features:
        feats = [str(i) for i in line]
        f.write(",".join(feats) + "\n")
