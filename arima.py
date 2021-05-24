import itertools
import warnings

import statsmodels.api as sm
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

p = d = q = range(0, 2)


df = pd.read_csv("data/20min-patterns.csv", nrows=1000)

df.plot.line(x='date_and_time', y='wave_angle')
pdq = list(itertools.product(p, d, q))
print(pdq)
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

y = df[["wave_angle"]]
y.set_index(df["date_and_time"])
warnings.filterwarnings("ignore") # specify to ignore warning messages
scaler = MinMaxScaler(feature_range=(0, 1))
y["wave_angle"] = scaler.fit_transform(y["wave_angle"].values.reshape(-1, 1))

plt.show()
# (0,0,3)(2,1,2)
mod = sm.tsa.SARIMAX(y, trend='n', order=(0, 0, 3), seasonal_order=(2, 1, 2, 12))
results = mod.fit()
print(results.summary())

df['forecast'] = results.predict(start=800, end=1000, dynamic=True)
df[['wave_angle', 'forecast']].plot(figsize=(12, 8))
plt.show()
