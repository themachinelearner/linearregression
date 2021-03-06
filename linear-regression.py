from sklearn import linear_model
import numpy as np

features = [[100, 2], [50, 42], [45, 31], [60, 35]]
labels = [5, 25, 22, 18]

lr = linear_model.LinearRegression()
lr = lr.fit(features, labels)
print (lr.coef_)
print (lr.intercept_)
print (lr.score(features, labels))
print (lr.predict([[100,2], [100, 20]]))