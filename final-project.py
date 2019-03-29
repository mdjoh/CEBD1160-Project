#!/usr/bin/env python

## import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sb

import plotly as py
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

## load standardized diabetes dataset and assign data to variables
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
corr_matrix = sb.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sb.diverging_palette(100, 220, as_cmap=True),
            square=True, ax=ax)
fig = corr_matrix.get_figure()
fig.savefig('diabetes_corr_matrix.png')
plt.clf()

## generate basic scatter plots to visualize Y vs. dataset feature
import os
import os.path as op

destination = str(os.getcwd()) + '/Simple Scatter Plots'
os.mkdir(destination)

for feature in headers:
    plt.scatter(df[feature], df[target_header])

    plt.xlabel(feature)
    plt.ylabel(target_header)

    plt.savefig(str(destination) + '/scatter_Y_vs_' + feature + '.png')
    plt.clf()

## Y vs All Features plot categorized by 'sex'
# split up categorical feature 'sex' into two separate datasets
sex1 = df.loc[df['sex'] < 0]
sex2 = df.loc[df['sex'] > 0]

trace0 = go.Scatter(
    x=sex1['age'], y=sex1['Y'],
    name='Age Sex 1',
    mode='markers',
    marker = {'color': '#7F3225', 'size': 10})

trace1 = go.Scatter(
    x=sex2['age'], y=sex2['Y'],
    name='Age Sex 2',
    mode='markers',
    marker={'color': '#7F3225', 'symbol': 'x', 'size': 10})

trace2 = go.Scatter(
    x=sex1['bmi'], y=sex1['Y'],
    name='BMI Sex 1',
    mode='markers',
    marker = {'color': '#F09E06', 'size': 10})

trace3 = go.Scatter(
    x=sex2['bmi'], y=sex2['Y'],
    name='BMI Sex 2',
    mode='markers',
    marker={'color': '#F09E06', 'symbol': 'x', 'size': 10})

trace4 = go.Scatter(
    x=sex1['bp'], y=sex1['Y'],
    name='BP Sex 1',
    mode='markers',
    marker = {'color': '#54F006', 'size': 10})

trace5 = go.Scatter(
    x=sex2['bp'], y=sex2['Y'],
    name='BP Sex 2',
    mode='markers',
    marker={'color': '#54F006', 'symbol': 'x', 'size': 10})

trace6 = go.Scatter(
    x=sex1['s1'], y=sex1['Y'],
    name='S1 Sex 1',
    mode='markers',
    marker = {'color': '#067BF0', 'size': 10})

trace7 = go.Scatter(
    x=sex2['s1'], y=sex2['Y'],
    name='S1 Sex 2',
    mode='markers',
    marker={'color': '#067BF0', 'symbol': 'x', 'size': 10})

trace8 = go.Scatter(
    x=sex1['s2'], y=sex1['Y'],
    name='S2 Sex 1',
    mode='markers',
    marker = {'color': '#8C06F0', 'size': 10})

trace9 = go.Scatter(
    x=sex2['s2'], y=sex2['Y'],
    name='S2 Sex 2',
    mode='markers',
    marker={'color': '#8C06F0', 'symbol': 'x', 'size': 10})

trace10 = go.Scatter(
    x=sex1['s3'], y=sex1['Y'],
    name='S3 Sex 1',
    mode='markers',
    marker = {'color': '#F006EC', 'size': 10})

trace11 = go.Scatter(
    x=sex2['s3'], y=sex2['Y'],
    name='S3 Sex 2',
    mode='markers',
    marker={'color': '#F006EC', 'symbol': 'x', 'size': 10})

trace12 = go.Scatter(
    x=sex1['s4'], y=sex1['Y'],
    name='S4 Sex 1',
    mode='markers',
    marker = {'color': '#DA778F', 'size': 10})

trace13 = go.Scatter(
    x=sex2['s4'], y=sex2['Y'],
    name='S4 Sex 2',
    mode='markers',
    marker={'color': '#DA778F', 'symbol': 'x', 'size': 10})

trace14 = go.Scatter(
    x=sex1['s5'], y=sex1['Y'],
    name='S5 Sex 1',
    mode='markers',
    marker = {'color': '#00DCFF', 'size': 10})

trace15 = go.Scatter(
    x=sex2['s5'], y=sex2['Y'],
    name='S5 Sex 2',
    mode='markers',
    marker={'color': '#00DCFF', 'symbol': 'x', 'size': 10})

trace16 = go.Scatter(
    x=sex1['s6'], y=sex1['Y'],
    name='S6 Sex 1',
    mode='markers',
    marker = {'color': '#241F2A', 'size': 10})

trace17 = go.Scatter(
    x=sex2['s6'], y=sex2['Y'],
    name='S6 Sex 2',
    mode='markers',
    marker={'color': '#241F2A', 'symbol': 'x', 'size': 10})

plot_data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9,
        trace10, trace11, trace12, trace13, trace14, trace15, trace16, trace17]

layout = go.Layout(
    xaxis = dict(title = 'Feature Scale', zeroline=False),
    yaxis= dict(title= 'Y', zeroline=False),
    showlegend = True,
    title='Y vs. All Features Plot')

fig = dict(data=plot_data, layout=layout)
plot(fig, filename='YvsAllFeatures.html')
# result: plot is not very informative; the data is randomly scattered

############# Dimensionality Reduction - PCA - to reduce redundancy among features ################

## drop categorical feature 'sex' in dataset
# standardized dataset:
nosex_df = df.drop(columns='sex')
nosex_data = np.delete(data, 1, axis=1)

# raw dataset:
# read and load raw data
file = 'diabetes_data.txt'

rawdata = pd.read_csv(file, sep='\s+|,' ,header=0, engine='python')
rawdf = pd.DataFrame(rawdata)

raw_nosex = np.delete(rawdata.values, [1,10], axis=1)

## generate cumulative explained variance curves
pca_std = PCA().fit(nosex_data)
pca_raw = PCA().fit(raw_nosex)

plt.plot(np.cumsum(pca_std.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.title('Cumulative Explained Variance of Standardized Data')
plt.savefig('StdExplainedVariance.png')
plt.clf()

plt.plot(np.cumsum(pca_raw.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.title('Cumulative Explained Variance of Raw Data')
plt.savefig('RawExplainedVariance.png')
plt.clf()

# split standardized and raw datasets into train and test sets
std_train, std_test, stdy_train, stdy_test = train_test_split(nosex_data, target, test_size = 0.3, random_state = 0)
raw_train, raw_test, rawy_train, rawy_test = train_test_split(raw_nosex, target, test_size = 0.3, random_state = 0)

# run PCA to determine the minimum number of components required to explain 90% of the variance of the dataset
pca_std = PCA(0.9).fit(std_train)
pca_raw = PCA(0.9).fit(raw_train)

# return "PCAed" data
trans_stdtrain = pca_std.transform(std_train)
trans_stdtest = pca_std.transform(std_test)
trans_rawtrain = pca_raw.transform(raw_train)
trans_rawtest = pca_raw.transform(raw_test)

# min number of components required to explain 90% variance
std_ncomponents = pca_std.n_components_ # returns: 6
raw_ncomponents = pca_raw.n_components_ # returns: 3

# raw data PC plot (example PCA visualization)
fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = trans_rawtrain[:,0] # PC 1
ydata = trans_rawtrain[:,1] # PC 2
zdata = trans_rawtrain[:,2] # PC 3
ax.scatter3D(xdata,ydata,zdata)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('The Three Principal Components (PC) from Raw Data without Sex Feature')
plt.savefig('PCAplot.png')
plt.clf()

############# Machine Learning - Linear Regression #################
## linear regession using raw data
raw_clf = LinearRegression().fit(raw_train, rawy_train)

predicted = raw_clf.predict(raw_test)
expected = rawy_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

raw_clf.score(raw_test, rawy_test)

## linear regression using standardized data
std_clf = LinearRegression().fit(std_train, stdy_train)

predicted = std_clf.predict(std_test)
expected = stdy_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

std_clf.score(std_test, stdy_test)

## linear regression using PCA on raw data
pcaraw_clf = LinearRegression().fit(trans_rawtrain, rawy_train)

predicted = pcaraw_clf.predict(trans_rawtest)
expected = rawy_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

pcaraw_clf.score(trans_rawtest, rawy_test)

## linear regression using PCA on standardized data
pcastd_clf = LinearRegression().fit(trans_stdtrain, stdy_train)

predicted = pcastd_clf.predict(trans_stdtest)
expected = stdy_test
print("RMS: %s" % np.sqrt(np.mean((predicted - expected)**2)))

pcastd_clf.score(trans_stdtest, stdy_test)
