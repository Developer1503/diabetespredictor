

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Importing the Diabetes Dataset
diabetes_dataset = pd.read_csv('C:/Users/VEDANT SHINDE/Downloads/archive/diabetes.csv')

# Printing the first 5 rows of the dataset
print(diabetes_dataset.head())

# Number of rows and columns in this dataset
print(diabetes_dataset.shape)

# Getting the statistical measures of the data
print(diabetes_dataset.describe())

# Value counts of 'Outcome' column
print(diabetes_dataset['Outcome'].value_counts())

# Mean values of features grouped by the outcome
print(diabetes_dataset.groupby('Outcome').mean())

# Data Standardization
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the SVM Model
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, Y_train)

# Training the Decision Tree Model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, Y_train)

# Training the Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Initializing with 100 trees
rf_classifier.fit(X_train, Y_train)

# Model Evaluation for SVM
svm_training_data_accuracy = accuracy_score(svm_classifier.predict(X_train), Y_train)
svm_test_data_accuracy = accuracy_score(svm_classifier.predict(X_test), Y_test)

# Model Evaluation for Decision Tree
dt_training_data_accuracy = accuracy_score(dt_classifier.predict(X_train), Y_train)
dt_test_data_accuracy = accuracy_score(dt_classifier.predict(X_test), Y_test)

# Model Evaluation for Random Forest
rf_training_data_accuracy = accuracy_score(rf_classifier.predict(X_train), Y_train)
rf_test_data_accuracy = accuracy_score(rf_classifier.predict(X_test), Y_test)

# Plotting the Accuracy Scores
models = ['SVM', 'Decision Tree', 'Random Forest']
training_accuracy = [svm_training_data_accuracy, dt_training_data_accuracy, rf_training_data_accuracy]
test_accuracy = [svm_test_data_accuracy, dt_test_data_accuracy, rf_test_data_accuracy]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_accuracy, width, label='Training Accuracy')
rects2 = ax.bar(x + width/2, test_accuracy, width, label='Test Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()

plt.show()

# Making a Predictive System using SVM
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
svm_prediction = svm_classifier.predict(std_data)

if svm_prediction[0] == 0:
    print('SVM Model Prediction: The person is not diabetic')
else:
    print('SVM Model Prediction: The person is diabetic')

# Making a Predictive System using Decision Tree
dt_prediction = dt_classifier.predict(std_data)

if dt_prediction[0] == 0:
    print('Decision Tree Model Prediction: The person is not diabetic')
else:
    print('Decision Tree Model Prediction: The person is diabetic')

# Making a Predictive System using Random Forest
rf_prediction = rf_classifier.predict(std_data)

if rf_prediction[0] == 0:
    print('Random Forest Model Prediction: The person is not diabetic')
else:
    print('Random Forest Model Prediction: The person is diabetic')
