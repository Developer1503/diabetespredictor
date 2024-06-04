Diabetes Prediction Models
This project demonstrates the use of machine learning algorithms to predict diabetes using the PIMA Indian Diabetes dataset. The following models are implemented and evaluated: Support Vector Machine (SVM), Decision Tree, and Random Forest.

Table of Contents
Installation
Usage
Dataset
Data Preprocessing
Model Training
Model Evaluation
Accuracy Comparison
Predictive Systems
Installation
Ensure you have Python and the necessary libraries installed. You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas scikit-learn matplotlib
Usage
To run the code, execute the script in your Python environment. Ensure the dataset path is correctly specified.

bash
Copy code
python diabetes_prediction.py
Dataset
The dataset used is the PIMA Indian Diabetes dataset. Download it from Kaggle.

Data Preprocessing
Load Dataset: The dataset is loaded using pandas.
Exploratory Data Analysis:
Display the first 5 rows.
Display the shape and statistical measures of the dataset.
Display the distribution of the 'Outcome' column.
Group the data by 'Outcome' and display the mean values.
Data Standardization: Standardize the feature values using StandardScaler.
Model Training
Split the Data: Split the standardized data into training and test sets.
Train Models: Train the following models:
Support Vector Machine (SVM)
Decision Tree
Random Forest (with 100 trees)
Model Evaluation
Evaluate each model using the accuracy score on both training and test datasets.
