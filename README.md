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

2. **Build a Linear Regression Model:**
   ``` python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   X = data.drop('MEDV', axis='columns')
   y = data['MEDV']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   lr = LinearRegression()
   lr.fit(X_train, y_train)

3. **Evaluate the Model:**
   ```python
   from sklearn import metrics
   y_pred = lr.predict(X_test)
   print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

4. **Examine Scaling:**
   ```python
   from sklearn.preprocessing import StandardScaler
   std_scaler = StandardScaler()
   std_scaler.fit(X_train)
   X_train_scaled = std_scaler.transform(X_train)
   X_test_scaled = std_scaler.transform(X_test)

5. ***L2 Regularization:**
   ```python
   from sklearn.linear_model import Ridge
   alpha_values = [0.01, 0.1, 1, 10, 100]
   for alpha in alpha_values:
       ridge_scale = Ridge(alpha=alpha).fit(X_train_scaled, y_train)

6. ***L1 Regularization:**
   ```python
   from sklearn.linear_model import Lasso
   for alpha in alpha_values:
       lasso_scale = Lasso(alpha=alpha).fit(X_train_scaled, y_train)

##Conclusion:
The analysis concludes that regularization methods do not significantly improve the model fit for this dataset.




## Problem 2: Classification with Breast Cancer Data

### Data Description
- **Title:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Creator:** Dr. William H. Wolberg, University of Wisconsin
- **Purpose:** To predict whether a tumor is malignant or benign.
- **Repository:** [Breast Cancer Wisconsin Dataset](https://raw.githubusercontent.com/sesillim/ai/main/BreastCancer.csv)

### Features
1. **radius_mean:** mean of distances from center to points on the perimeter
2. **texture_mean:** standard deviation of gray-scale values
3. **perimeter_mean:** mean size of the core tumor
4. **area_mean:** mean area of the tumor
5. **smoothness_mean:** mean of local variation in radius lengths
6. **compactness_mean:** mean of perimeter^2 / area - 1.0
7. **concavity_mean:** mean of severity of concave portions of the contour
8. **concave points_mean:** mean for number of concave portions of the contour
9. **symmetry_mean:** mean of symmetry
10. **fractal_dimension_mean:** mean for "coastline approximation" - 1

### Target
- **diagnosis:** The diagnosis of the tumor (M = malignant, B = benign)

### Tasks
1. **Count the number of observations in the data.**
2. **Build a logistic regression model. Print the coefficients and assess the fit.**
3. **Examine whether scaling improves the fit of the logistic regression model.**
4. **Implement L2 regularization (Ridge Classifier) with scaling and compare different hyperparameter values.**
5. **Implement L1 regularization (Lasso Classifier) with scaling and compare different hyperparameter values.**

### Implementation

1. **Load and Explore the Data:**
   ```python
   import pandas as pd
   import numpy as np
   url = 'https://raw.githubusercontent.com/sesillim/ai/main/BreastCancer.csv'
   data = pd.read_csv(url)
   data.describe()

2. **Count the Number of Observations:**
   ```python
   print("The number of observations are", data.shape[0])

3. **Build a Logistic Regression Model:**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   X = data.drop('diagnosis', axis='columns')
   y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   lr = LogisticRegression()
   lr.fit(X_train, y_train)

4. **Evaluate the Model:**
   ```python
   from sklearn import metrics
   y_pred = lr.predict(X_test)
   print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

5. **Examine Scaling:**
   ```python
   from sklearn.preprocessing import StandardScaler
   std_scaler = StandardScaler()
   std_scaler.fit(X_train)
   X_train_scaled = std_scaler.transform(X_train)
   X_test_scaled = std_scaler.transform(X_test)

6. **L2 Regularization (Ridge Classifier):**
   ```python
   from sklearn.linear_model import RidgeClassifier
   alpha_values = [0.01, 0.1, 1, 10, 100]
   for alpha in alpha_values:
       ridge = RidgeClassifier(alpha=alpha)
       ridge.fit(X_train_scaled, y_train)
       y_pred_ridge = ridge.predict(X_test_scaled)
       print(f'Alpha: {alpha}, Accuracy: {metrics.accuracy_score(y_test, y_pred_ridge)}')

7. **L1 Regularization (Lasso Classifier):**
   ```python
   from sklearn.linear_model import Lasso
   for alpha in alpha_values:
       lasso = Lasso(alpha=alpha)
       lasso.fit(X_train_scaled, y_train)
       y_pred_lasso = lasso.predict(X_test_scaled)
       y_pred_lasso_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_lasso]
       print(f'Alpha: {alpha}, Accuracy: {metrics.accuracy_score(y_test, y_pred_lasso_binary)}')

##Conclusion:
The analysis concludes that scaling and regularization methods didn't help improve the accuracy of the logistic regression model for this dataset. Different values of alpha in L2 and L1 regularization were tested to find the best model fit.








