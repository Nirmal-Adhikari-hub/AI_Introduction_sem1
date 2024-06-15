# AI Introduction Semester 1 Projects

This repository contains solutions for two machine learning problems, focused on regression and classification tasks using Python and scikit-learn. Below are detailed descriptions of each problem, the data used, and the steps taken to solve them.

## Problem 1: Boston Housing Data

### Data Description
- **Title:** Boston housing data
- **Creator:** Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.
- **Purpose:** To predict housing prices in the Boston area.
- **Repository:** [Boston Housing Dataset](https://raw.githubusercontent.com/sesillim/ai/main/Housing.csv)

### Features
1. **CRIM:** per capita crime rate by town
2. **ZN:** proportion of residential land zoned for lots over 25,000 sq.ft.
3. **INDUS:** proportion of non-retail business acres per town
4. **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
5. **NOX:** nitric oxides concentration (parts per 10 million)
6. **RM:** average number of rooms per dwelling
7. **AGE:** proportion of owner-occupied units built prior to 1940
8. **DIS:** weighted distances to five Boston employment centres
9. **RAD:** index of accessibility to radial highways
10. **TAX:** full-value property-tax rate per $10,000
11. **PTRATIO:** pupil-teacher ratio by town
12. **B:** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13. **LSTAT:** % lower status of the population

### Target
- **MEDV:** Median housing price value of owner-occupied homes in $1000's

### Tasks
1. **Count the number of observations in the data.**
2. **Build a linear regression model. Print the coefficients and assess the fit.**
3. **Examine whether scaling improves the fit of the linear regression model.**
4. **Implement L2 regularization (Ridge Regression) with scaling and compare different hyperparameter values.**
5. **Implement L1 regularization (Lasso Regression) with scaling and compare different hyperparameter values.**

### Implementation

1. **Load and Explore the Data:**
   ```python
   import pandas as pd
   import numpy as np
   url = 'https://raw.githubusercontent.com/sesillim/ai/main/Housing.csv'
   data = pd.read_csv(url)
   data.describe()
