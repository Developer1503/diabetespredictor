# Diabetes Prediction using Machine Learning

## Overview
This project implements a **machine learning model** to predict diabetes using patient health data. It explores different classification algorithms and evaluates their performance.

## Dataset
The dataset used is **diabetes.csv**, which contains the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (1 = Diabetic, 0 = Not Diabetic)

## Technologies Used
- **Python**
- **Jupyter Notebook**
- **NumPy, Pandas** for data manipulation
- **Scikit-learn** for machine learning models
- **Matplotlib** for visualization

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   ```
2. Navigate to the project folder:
   ```bash
   cd diabetes-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and execute the notebook file `diabetes_prediction.ipynb` step by step.

## Models Implemented
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**
- **Random Forest Classifier**

## Performance Evaluation
The models are evaluated using **accuracy scores** for training and test sets.

## Results Visualization
A bar chart is plotted to compare model performance.

## Prediction Example
The trained models can predict whether a patient has diabetes based on input features. Example:
```python
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_numpy_array)
prediction = model.predict(std_data)
```

## Contributing
Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the **MIT License**.
