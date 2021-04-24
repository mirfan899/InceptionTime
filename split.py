import pandas as pd
test = pd.read_csv("features200.csv", nrows=10000)
test.to_csv("Wave_TEST.csv")
test.to_csv("Wave_TEST.csv", index=False)
data = pd.read_csv("features200.csv")
train = data.head(40000)
train.to_csv("Wave_TRAIN.csv", index=False, header=False)
test = data.tail(10000)
test.to_csv("Wave_TEST.csv", index=False, header=False)