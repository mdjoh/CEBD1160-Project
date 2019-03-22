#!/usr/bin/env python

## import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.datasets import load_diabetes # repurposing plotting code week 8 homework
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

## load diabetes dataset and assign data to variables
diabetes = load_diabetes(return_X_y = False)

data = diabetes.data
headers = diabetes.feature_names
target = diabetes.target
target_header = 'Y'

## make a dataframe for correlation matrix and plotting
# reshape target array
reshaped_target = target.reshape(len(target),1)

df = pd.DataFrame(data=np.append(data, reshaped_target, axis=1), columns=headers+[target_header])

## correlation martrix
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sb.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sb.diverging_palette(100, 220, as_cmap=True),
            square=True, ax=ax)

## main visualization plot
# Y will be on z-axis
# categorize by sex and colour code it

############# Dimensionality Reduction - PCA - to reduce redundancy among features ################
X=data
y=target

pca = PCA(n_components=3, whiten=False)
pca.fit(X)

# explained variance plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

print(pca.components_)
print(pca.explained_variance_)

# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

############# Regression #################
# Split the data and target into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target)

### Linear Regression ###
clf = LinearRegression()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

# print the model score
clf.score(X_test, y_test)

# plot predicted vs expected
plt.scatter(expected, predicted)
plt.xlabel('Features')
plt.ylabel('Predicted ' + target_header)
plt.savefig('LinearRegOutput.png')
plt.clf()
