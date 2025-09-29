# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
              import pandas as pd
              import numpy as np
              from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler
              from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
              from sklearn.linear_model import LogisticRegression
              
              # Load dataset
              df = pd.read_csv("/content/bmi.csv")  
              df.head()

<img width="387" height="216" alt="image" src="https://github.com/user-attachments/assets/ba8f73c8-6c33-484f-8745-0c91fb0556c6" />

              df = df.dropna()  # Drop missing values


              scaler = MinMaxScaler()
              df_scaled_minmax = scaler.fit_transform(df[['Height','Weight']])
              df_scaled_minmax = pd.DataFrame(df_scaled_minmax, columns=['Height','Weight'])
              df_scaled_minmax.head()

<img width="514" height="223" alt="image" src="https://github.com/user-attachments/assets/623858d2-e830-4177-9a4e-089d8b90fcca" />

              scaler = StandardScaler()
              df_scaled_standard = scaler.fit_transform(df[['Height','Weight']])
              df_scaled_standard = pd.DataFrame(df_scaled_standard, columns=['Height','Weight'])
              df_scaled_standard.head()

<img width="283" height="232" alt="image" src="https://github.com/user-attachments/assets/e0a3f464-cfcf-4b46-b11e-d33d1959caa9" />

              scaler = Normalizer()
              df_scaled_norm = scaler.fit_transform(df[['Height','Weight']])
              df_scaled_norm = pd.DataFrame(df_scaled_norm, columns=['Height','Weight'])
              df_scaled_norm.head()

<img width="295" height="237" alt="image" src="https://github.com/user-attachments/assets/c9813a86-2f6c-4939-9421-67a4914c0ece" />

              scaler = MaxAbsScaler()
              df_scaled_maxabs = scaler.fit_transform(df[['Height','Weight']])
              df_scaled_maxabs = pd.DataFrame(df_scaled_maxabs, columns=['Height','Weight'])
              df_scaled_maxabs.head()

<img width="328" height="230" alt="image" src="https://github.com/user-attachments/assets/92c3edc0-2fa9-4a2c-92cf-d5f36981522b" />


              scaler = RobustScaler()
              df_scaled_robust = scaler.fit_transform(df[['Height','Weight']])
              df_scaled_robust = pd.DataFrame(df_scaled_robust, columns=['Height','Weight'])
              df_scaled_robust.head()

<img width="308" height="220" alt="image" src="https://github.com/user-attachments/assets/d8be7019-bad5-47ea-b14e-935cf51db4bf" />

              #filter
              from sklearn.feature_selection import SelectKBest, mutual_info_classif
              
              # Features and target
              X = df[['Height','Weight']]
              y = df['Index']
              
              # Apply Filter Method
              filter_selector = SelectKBest(score_func=mutual_info_classif, k=2)
              X_filter = filter_selector.fit_transform(X, y)
              
              selected_filter_features = X.columns[filter_selector.get_support()]
              print("Filter Method Selected Feature(s):", selected_filter_features)

<img width="800" height="58" alt="image" src="https://github.com/user-attachments/assets/67420a3d-d651-4373-ae82-1a1c6b932105" />


              #wrapper
              from sklearn.feature_selection import RFE
              from sklearn.linear_model import LogisticRegression
              from sklearn.preprocessing import StandardScaler
              
              # Scale the features first
              scaler = StandardScaler()
              X_scaled = scaler.fit_transform(X)
              
              # Increase max_iter to ensure convergence
              model = LogisticRegression(max_iter=1000)
              wrapper_selector = RFE(model, n_features_to_select=1)
              wrapper_selector.fit(X_scaled, y)
              
              selected_wrapper_features = X.columns[wrapper_selector.support_]
              print("Wrapper Method Selected Feature(s):", selected_wrapper_features)

<img width="720" height="60" alt="image" src="https://github.com/user-attachments/assets/c683efd3-72f8-4bfa-be30-64aaeab3ef93" />


              #embedded
              from sklearn.linear_model import Lasso
              
              lasso = Lasso(alpha=0.01)
              lasso.fit(X, y)
              
              # Check which coefficients are non-zero
              selected_embedded_features = X.columns[lasso.coef_ != 0]
              print("Embedded Method Selected Feature(s):", selected_embedded_features)

<img width="894" height="94" alt="image" src="https://github.com/user-attachments/assets/be686aa9-9c30-4c57-990f-e778730031e0" />




# RESULT:
      Thus the expected output is achieved.
