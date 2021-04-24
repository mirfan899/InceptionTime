import pandas as pd
import numpy as np

WAVE_TYPE = {"down": 0, "up": 1}
data = pd.read_csv("180min-patterns.csv")
total_groups = data["wave_grp_id"].max()

total_features = []
max_count = []
# for grp in range(len(total_groups)):
for grp in range(total_groups):
    features = []
    rows = data[data["wave_grp_id"] == grp]
    max_count.append(len(rows.index))
    records = len(rows)
    difference = records-100
    if difference >= 0:
        for i in rows.index[:20]:
            features.append(rows["cp"][i])
            features.append(rows["kvFst"][i])
            features.append(rows["kvTrgger"][i])
            features.append(rows["ripple_angle"][i])
            features.append(rows["wave_angle"][i])
    else:
        for i in rows.index[:20]:
            features.append(rows["cp"][i])
            features.append(rows["kvFst"][i])
            features.append(rows["kvTrgger"][i])
            features.append(rows["ripple_angle"][i])
            features.append(rows["wave_angle"][i])
        for i in range(len(features)+1, 101):
            features.append(0.0)
        print(len(features))
    features.append(WAVE_TYPE[rows["wave_type"][rows.index[0]]])
    total_features.append(features)

print(np.array(total_features).shape)
print(len(total_features))
print(len(total_features[3]))
# print(sorted(max_count))
print(max(max_count))
print(min(max_count))
with open("features.csv", "w") as f:
    for line in total_features:
        feats = [str(i) for i in line]
        f.write(",".join(feats)+"\n")
