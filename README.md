# Diabetes Predictions
## Profile
Nabila Aisha - Business Statistics

## About
The diabetes_prediction_dataset.csv file contains medical and demographic data of patients along with their diabetes status, whether positive or negative. It consists of various features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. The Dataset can be utilized to construct machine learning models that can predict the likelihood of diabetes in patients based on their medical history and demographic details.

## Cases
To predict diabetes patients status

## Analysis
### A. Data Understanding
```
from google.colab import files
uploaded = files.upload()
import io
import pandas as pd

data = pd.read_excel(io.BytesIO(uploaded['diabetes_prediction_dataset.xlsx']))
```
Function : To import dataset in google colab, named as 'data'

### B. Data Preparation
#### 1 . Exploratory Data Analysis
```
#1.1 Feature Checking
titles = ['diabetes']
datasets = [data]

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
features_check = pd.DataFrame({}, )
features_check['datasets'] = titles

features_check['numeric_features'] = [len((df.select_dtypes(include = numerics)).columns) for df in datasets]
features_check['numerical_features_name'] = [', '.join(list((df.select_dtypes(include = numerics)).columns)) for df in datasets]
features_check['categorical_features'] = [len((df.select_dtypes(include = 'object')).columns) for df in datasets]
features_check['categorical_features_name'] = [', '.join(list((df.select_dtypes(include = 'object')).columns)) for df in datasets]
features_check['boolean_features'] = [len((df.select_dtypes(include = 'bool')).columns) for df in datasets]
features_check['boolean_features_name'] = [', '.join(list((df.select_dtypes(include = 'bool')).columns)) for df in datasets]
features_check['total_columns'] = [len(df.columns) for df in datasets]
features_check['total_rows'] = [len(df) for df in datasets]
features_check.style.background_gradient(cmap = 'Blues')
```
Function = To check feature names and status (categorical and numerical status)

```
#1.2 Missing values checking
missing_check = pd.DataFrame({}, )
missing_check['datasets'] = titles
missing_check['features'] = [', '.join([col for col, null in df.isnull().sum().items()]) for df in datasets]
missing_check['null_amount'] = [df.isnull().sum().sum() for df in datasets]
missing_check['null_features_amount'] = [len([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]
missing_check['null_features'] = [', '.join([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]
missing_check.style.background_gradient(cmap='Blues')
```
Function = to check if there is any missing values in variables

```
#1.3 Automated EDA
!pip install dataprep
from dataprep.eda import create_report
create_report(data).show()
```
Function = to create automated EDA that contains information about the features, histogram, duplicates, missing values, correlation and so on

### C. Data Exploration
#### 1 . Feature Engineering
```
# Change categorical to numeric
from sklearn.preprocessing import LabelEncoder

dataset = ['hypertension', 'heart_disease', 'diabetes', 'gender', 'smoking_history']
for col in dataset:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
data.head()
# Handle categorical variables
df = pd.get_dummies(data, columns=['gender', 'smoking_history','hypertension','heart_disease'])
df.head()
```
Function = Analyze can be done if the features are set into the same types. there are numerical features and categorical features. So, this code is to change categorical features into numerical features

### D. Modelling
#### 1. Splitting Data
```
from sklearn.model_selection import train_test_split
# Split the data
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Function = splitting data into 80% train - 20% test. Train data is used to make models, and test data is used to make prediction

#### 2. Multilayer Perceptron
```
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
```
Function = analyze the data using Multilayer perceptron (MLP) using logistic activation function

### 3. Predictions Data
```
y_prob = model.predict_proba(X_test)
threshold = 0.5
y_pred_binary = (y_prob[:, 1] > threshold).astype(int)
print("Binary Predictions:")
print(y_pred_binary)
```
Function = predictions values type is decimal. But, Y variables are categoric, contains 1 and 0 score. So, this code is to set prediction values into 1 and 0 scores

```
from sklearn.metrics import accuracy_score

print("Binary Predictions:")
print(y_pred_binary)

accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')

```
Function =  To analyze the accuracy score

### 4. Graphic True and False Predict
```
import matplotlib.pyplot as plt
import numpy as np

# Assuming y_test is the true labels and y_pred_binary is the predicted labels

# Count the number of correct predictions
correct_predictions = np.sum(y_test.values == y_pred_binary)

# Count the number of incorrect predictions
incorrect_predictions = len(y_test) - correct_predictions

# Plot the bar chart
labels = ['Correct Predictions', 'Incorrect Predictions']
values = [correct_predictions, incorrect_predictions]

plt.bar(labels, values, color=['green', 'red'])
plt.title('Correct vs Incorrect Predictions')
plt.ylabel('Number of Samples')

# Add labels to each bar
for i in range(len(labels)):
    plt.text(i, values[i] + 0.1, str(values[i]), ha='center', va='bottom')

plt.show()
```
Function = to make bar chart that contains true and false predict values

