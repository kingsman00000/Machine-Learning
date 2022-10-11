# Train a logistic regression classifier to predict whether a flower is iris virginica or not ?

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])      
                    #   describes the data and the target corressponding to the data

X = iris["data"][:, 3:]

Y = (iris["target"] == 2).astype(np.int)  # here we took == 2 to say the flower is iris virginica and it is true or false
# print(X)
# print(Y)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X,Y)
example = clf.predict(([[2.6]]))
print(example)

# Using Matplotlib to plot the  visualization
X_new = np.linspace(0,3,1000).reshape(-1,1)
#print(X_new)
Y_prob = clf.predict_proba(X_new)
#print(Y_prob)
plt.plot(X_new, Y_prob[:,1], "g-", label="virginica")
plt.show()


