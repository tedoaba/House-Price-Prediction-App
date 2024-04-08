import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# House Price Prediction App

This app predicts the **Boston House Price** based on various features.
""")

# Dataset Information and How to Use
st.write("""
### Dataset Information

This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston, MA. 
It includes various features such as per capita crime rate by town, average number of rooms per dwelling, 
proportion of non-retail business acres per town, and more.

You can adjust the input parameters from the sidebar to predict the median value of owner-occupied homes (MEDV) in Boston.

Dataset Link: http://lib.stat.cmu.edu/datasets/boston
""")

# Load the Boston housing dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
Y = pd.DataFrame(target, columns=["MEDV"])

# Display first 5 elements of the dataset
st.write("## First 5 Elements of the Dataset")
st.write(X.head())

# Explanation of each column
st.write("""
#### Explanation of Each Column

- CRIM: Per capita crime rate by town.
- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: Proportion of non-retail business acres per town.
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- NOX: Nitric oxides concentration (parts per 10 million).
- RM: Average number of rooms per dwelling.
- AGE: Proportion of owner-occupied units built prior to 1940.
- DIS: Weighted distances to five Boston employment centers.
- RAD: Index of accessibility to radial highways.
- TAX: Full-value property tax rate per $10,000.
- PTRATIO: Pupil-teacher ratio by town.
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of [people of African American descent] by town.
- LSTAT: Percentage of lower status of the population.
- MEDV: Median value of owner-occupied homes in $1000s.
""")

# Calculate correlation between features and targets
correlation = X.join(Y).corr()

# Plot histogram for correlation
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("### Correlation between Features and Target (MEDV)")
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
st.pyplot()

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Select Input Parameter Values')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
    TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
    PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
    LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Selected Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write("Here is the predicted median value of owner-occupied homes (MEDV) is:")
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# Explaining the model's predictions using SHAP values
st.write("""
### Explanation of Prediction using SHAP Values

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. 
It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

SHAP values represent the impact of each feature on the model's output. 
A positive SHAP value for a feature means the feature pushes the prediction higher, while a negative SHAP value means the feature pushes the prediction lower.

See more: https://github.com/slundberg/shap

Here are the SHAP values for each feature:
""")
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# Ignore plot warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
### Feature Importance

Feature importance shows the most important features in the model for making predictions. 
It helps to understand which features have the most influence on the target variable.

Below are the feature importance scores based on SHAP values:
""")
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')


st.write("""
##### Thank you!

**Contributor:** Tadesse Abateneh

**GitHub:** https://github.com/tedoaba

**LinkedIn:** https://www.linkedin.com/in/tadesse-abateneh/
""")