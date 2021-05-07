import pandas as pd

WAVE_TYPE = {"down": 0, "up": 1}
INVERSE_WAVE_TYPE = {0: "down", 1: "up"}
data = pd.read_csv("data/180min-patterns.csv")
data["wave_type"] = data["wave_type"].map({"down": 0, "up": 1})

total_features = []
cp = []
kvfst = []
kvtrgger = []
ripple_angle = []
wave_angle = []
wave_type = []

for i in data.iterrows():
    cp.append(i[1]["cp"])
    kvfst.append(i[1]["kvFst"])
    kvtrgger.append(i[1]["kvTrgger"])
    ripple_angle.append(i[1]["ripple_angle"])
    wave_angle.append(i[1]["wave_angle"])
    wave_type.append(i[1]["wave_type"])


total_features = pd.DataFrame({"cp": cp, "kvFst":kvfst, "kvTrgger":kvtrgger, "ripple_angle": ripple_angle,"wave_angle":wave_angle, "wave_type":wave_type})

total_features.to_csv("features/WaveByRow.csv", index=False, header=False)