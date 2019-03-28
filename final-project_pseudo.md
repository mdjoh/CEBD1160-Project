Convert np array to pd dataframe

# Visualization
scatter plots of Y vs. features to see how data are distributed
Normalize, stratify by sex 2D scatter plot (Y vs. normalized feature score)

# Constructing model
run PCA on unstandardized data and standardized data (no categorical features)
determine minimum number of PCs needed to explain 90% of variance (std: 6, raw: 3)
visualize principal components (PCA plot with raw data)

run a regression model (pick method based on how components look on a plot)
k nearest regression because the data appears to be randomly distributed and no strong correlations between the features and Y are apparent

compare raw, raw_pca, std_pca model scores

Do linear regression - it's simple
machine learning - split dataset into training and testing sets
train and test model
Evaluate model

PCA was used to make the simplest prediction model possible. Reducing the dimensionality of the dataset was hypothesized to enable a simpler prediction model to be built. PCA was conducted to explain 90% of the variance in the original dataset. Thus, this process was believed to result in a simpler prediction model that would only sacrifice prediction ability to a slight degree.

With the goal of constructing the simplest possible model (hence dimensionality reduction using PCA), linear regression modeling technique was used.

Model performance scores indicate that the diabetes progression prediction models made for this project can be improved. To improve prediction ability, other regression techniques can be explored to construct prediction models that would yield better prediction performance. Multiple techniques can be explored and their performances evaluated to determine which one performs best when predicting diabetes disease progression (Y).
Linear regression was likely not the best regression technique to use for this dataset.
A future direction can be to use another regression technique that better fits this dataset or try multiple techniques and evaluate which one performs the best

Table of results (RMSE) in markdown?
Regression with principal components to predict Y show marginal improvements in model performance score

PCA regression improved model scores from non-PCA regression as expected for standardized data but not for raw data

Model Score, RMS
Raw data: 0.3594, 57.1664
Standardized data: 0.3593, 57.1667
PCA on raw data: 0.2766, 60.7500
PCA on standardized data: 0.3649, 56.9196
