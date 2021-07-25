import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# data loader
X, y = load_iris(return_X_y=True)
# print(X)
clf = LogisticRegression(random_state=0).fit(X, y)
test = clf.predict(X[:2, :])
print(test)