import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plot

frame = pd.read_csv('breastcancer.data')
frame.replace('?',-99999,inplace=True)
frame.drop(['code'],1,inplace=True)


labels = np.array(frame.drop(['class'],1))
classes = np.array(frame['class'])

X_train, x_test, y_train, y_test = train_test_split(labels,classes)

nearestN = neighbors.KNeighborsClassifier()
nearestN.fit(X_train,y_train)

plot.scatter(frame.iloc[0],frame.iloc[-1])
plot.show()

sample = np.array([4,2,1,1,1,2,3,2,1])
sample = sample.reshape(1,-1)
prediction = nearestN.predict(sample)
print(prediction)