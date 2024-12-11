# Project Machine Learning App

Project Overview :

This Streamlit application enables users to explore, transform, analyze, and predict data through an interactive interface.

### App Structure

The app is organized into three main pages:

1. Data Loading: Upload and preview datasets.
2. Data Management: Clean and preprocess data.
3. Machine Learning: Train and evaluate models.

Navigation between pages is facilitated via a sidebar with radio buttons.

### Data Overview Examples

#### Diabetes Dataset:

![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/fe9bf587-2e2e-45b5-9c06-98f4f35b8dac)

#### Wine Dataset:

![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/ccf14973-c13e-4694-a26c-a2fda2fdce19)

### Data Processing and Analysis
#### Data Cleaning: 
Detect missing values, empty columns, and provide options to:
- Select and delete empty columns.
- Replace missing values with mean, median, or custom values.
- Change variable types.

#### Descriptive Analysis: 
Provides summary statistics (mean, standard deviation, median, quartiles, etc.).

Example for Wine Dataset:
![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/dd4b2060-6cbb-4504-bc0e-b5d485654462)

##### Visualizations :
Histogram plots to visualize data distribution.

Example:
![image](https://github.com/diginamic-formation/projet-machine-learning/assets/75080561/b8e73afa-993c-4c88-a4ea-66286edad574)

#### Correlation Analysis:
Identify variables most correlated with the target variable.

#### Data Standardization: 
Standardize data to homogenize data scales.


### Machine Learning Pipeline

#### Algorithm Selection: 
Choose algorithms based on the problem type (classification or regression).
#### Data Splitting: 
Automatically split data into training and testing sets.
#### Model Training: 
Train the selected model and display progress.
#### Prediction: 
Predict on test data or new user-provided data.

### Model Evaluation

#### Regression Models:
- Accuracy Metrics: Display R², MSE, MAE.
- Visualizations:  
  - Plot actual vs. predicted values.
  - Plot MSE against the alpha parameter.
- Model Comparison:
  - Use K-Fold cross-validation to evaluate models.
  - Calculate R², MSE, MAE for each model.
  - Select the best model based on average R².


#### Classification Models:
- Accuracy Metrics: Display accuracy, precision, recall.
- Visualizations: Generate confusion matrix.
- Model Comparison:
  - Use K-Fold cross-validation to evaluate models.
  - Calculate precision, recall, and F1-score for each model.
  - Compare average scores and select the best model based on these metrics.
