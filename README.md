# cebd1160_project_template

| Name | Date |
|:-------|:---------------|
|Marchiano Dong Jun Oh | March 29, 2019|

-----

### Resources
Your repository should include the following:

- Python script for your analysis
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question

What is the predicted progression of diabetes given age, sex, BMI, blood pressure, S1, S2, S3, S4, S5, and S6 features?

### Abstract

The diabetes dataset presents ten measured features to track the progression of diabetes for 442 subjects. The progression is also presented as a feature called 'Y'. The task at hand is to be able to predict the progression of diabetes (i.e. Y) by constructing a model using machine learning from the other features in the dataset. Python data manipulation, data visualization, and machine learning libraries (e.g. numpy, pandas, matplotlib, seaborn, scikit-learn) will be used to create the data model that would predict Y. A python script to construct the machine learning model from this dataset that would predict Y was written. Plots were made for exploratory purposes to visualize how the data were distributed and if any trends were apparent. Visualization was also used to determine which features are best to include in the model. Machine learning model creating functions in the scikit-learn library were used to create the 'Y' predicting model.

### Introduction

The diabetes dataset consists of ten baseline variables-age, sex, body mass index, average blood pressure, and six blood serum measurements- that were obtained for each of 442 diabetes patients and the response of interest, a quantitative measure of disease progression one year after baseline ('Y') [1].

### Methods

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)
- why you chose this method

### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
- At least 1 "value" that summarizes either your data or the "performance" of your method
- A short explanation of both of the above

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
[1]. Adapted from https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset

-------
