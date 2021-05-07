import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np


def readucr(filename):
    data = np.loadtxt(filename, delimiter=",")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "features/WaveByRow.csv"

X, Y = readucr(root_url)

# Feel free to use different ratios to split the data.
train_text, test_text, train_labels, test_labels = train_test_split(X, Y, test_size=0.30, random_state=42)


# train model
clf = LogisticRegression(max_iter=100).fit(train_text, train_labels)

# test model
test_pred = clf.predict(test_text)

acc = accuracy_score(test_labels, test_pred)
pre, rec, f1, _ = precision_recall_fscore_support(test_labels, test_pred, average='macro')
print('acc', acc)
print('precision', pre)
print('rec', rec)
print('f1', f1)

# save model and other necessary modules
all_info_want_to_save = {
    'model': clf,
}
save_path = open("models/lr_wave.pickle", "wb")
pickle.dump(all_info_want_to_save, save_path)
