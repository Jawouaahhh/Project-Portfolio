import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC


def validation_k_fold_classification(X, y, FEATURES, model):
    """
    Performs K-Fold cross-validation to evaluate a classification model.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Class labels.
        FEATURES (list): List of feature names to use for training.
        model (object): Classification model to evaluate.

    Returns:
        dict: Dictionary with average accuracy, precision, recall, and F1-score.
    """
    kf = KFold(n_splits=6, shuffle=True, random_state=2021)  # Initialize K-Fold cross-validator
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}  # Dictionary to store scores
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Set up plot for confusion matrices

    X = pd.DataFrame(X, columns=FEATURES)  # Ensure X is a DataFrame
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        ligne = i // 3  # Row index for subplot
        colonne = i % 3  # Column index for subplot

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Split data into training and test sets
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train[FEATURES], y_train)  # Train the model
        y_pred = model.predict(X_test[FEATURES])  # Predict on the test set

        # Compute scores for current fold
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, average='weighted'))
        scores["recall"].append(recall_score(y_test, y_pred, average='weighted'))
        scores["f1"].append(f1_score(y_test, y_pred, average='weighted'))

        # Plot confusion matrix
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Reds", ax=axes[ligne, colonne])
        axes[ligne, colonne].set_title(f'Fold {i + 1}')

    st.subheader("Model Performance")
    st.pyplot(fig)  # Display plot in Streamlit

    # Calculate mean scores across all folds
    mean_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}

    st.subheader("Average Model Validation Metrics")
    mean_scores_df = pd.DataFrame.from_dict(mean_scores, orient='index', columns=['Mean Score'])
    st.table(mean_scores_df)

    return mean_scores


def validation_k_fold_regression(X, y, FEATURES, model):
    """
    Performs K-Fold cross-validation on a regression model and displays results.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        FEATURES (list): List of feature names.
        model: Regression model instance.

    Returns:
        dict: Mean scores for R2, MSE, and MAE across all folds.
    """
    kf = KFold(n_splits=6, shuffle=True, random_state=2021)  # Initialize K-Fold cross-validator
    columns = ["R2", "MSE", "MAE"]  # Metrics to track
    stats_df = pd.DataFrame(columns=columns)  # DataFrame to store metrics
    lignes = 2
    colonnes = 3
    fig, axes = plt.subplots(lignes, colonnes, figsize=(15, 10))  # Set up plot for results

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        ligne = i // colonnes  # Row index for subplot
        colonne = i % colonnes  # Column index for subplot
        X = pd.DataFrame(X, columns=FEATURES)  # Ensure X is a DataFrame
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Split data into training and test sets
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train[FEATURES], y_train)  # Train the model
        y_pred = model.predict(X_test[FEATURES])  # Predict on the test set

        # Compute metrics for current fold
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        stats_df.loc[i] = [round(r2, 4), round(mse, 4), round(mae, 4)]

        # Plot scatter plot
        axes[ligne, colonne].scatter(y_test, y_pred)
        axes[ligne, colonne].plot(np.arange(y_test.min(), y_test.max()), np.arange(y_test.min(), y_test.max()),
                                  color='red', linestyle='--')  # Reference line

        axes[ligne, colonne].set_title(f'R2: {round(r2, 4)} - MSE: {round(mse, 4)} - MAE: {round(mae, 4)}')
        axes[ligne, colonne].set_xlabel("Actual Values")
        axes[ligne, colonne].set_ylabel("Predicted Values")

    mean_r2 = stats_df["R2"].mean()
    mean_mse = stats_df["MSE"].mean()
    mean_mae = stats_df["MAE"].mean()

    # Display results in Streamlit
    st.subheader("Model Performance")
    st.pyplot(fig)

    st.subheader("Average Model Validation Metrics")
    stats_mean_df = stats_df.mean().to_frame().reset_index()
    stats_mean_df = stats_mean_df.rename(columns={'index': 'Metric', 0: 'Mean'})
    st.table(stats_mean_df)

    return {"R2": mean_r2, "MSE": mean_mse, "MAE": mean_mae}


def compare_regression_models(X, y, FEATURES):
    """
    Compares multiple regression models using K-Fold cross-validation and selects the best model.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        FEATURES (list): List of feature names.

    Returns:
        tuple: Name of the best model and the best model instance.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42)
    }

    results = {}

    st.header("Regression Model Comparison")

    for model_name, model in models.items():
        st.subheader(model_name)
        scores = validation_k_fold_regression(X, y, FEATURES, model)  # Validate model
        results[model_name] = scores

    # Select the best model based on the highest R2 score
    best_model_name = max(results, key=lambda x: results[x]["R2"])
    st.subheader(f"Best Model: {best_model_name} with an average R2 of {results[best_model_name]['R2']:.4f}")

    return best_model_name, models[best_model_name]


def compare_classification_models(X, y, FEATURES):
    """
    Compares multiple classification models and selects the best model based on average accuracy, precision, recall, and F1-score.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Class labels.
        FEATURES (list): List of feature names to use for training.

    Returns:
        tuple: Name of the best model and the best model instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Tree Classifier": DecisionTreeClassifier(random_state=42),
        "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=3),
        "Support Vector Classifier": SVC(probability=True),
        "Naive Bayes Classifier": GaussianNB(),
        "XGBoost Classifier": XGBClassifier(random_state=42),
        "Stochastic Gradient Descent Classifier": SGDClassifier(random_state=42)
    }

    results = {}

    st.header("Classification Model Comparison")

    for model_name, model in models.items():
        st.subheader(model_name)
        mean_scores = validation_k_fold_classification(X, y, FEATURES, model)  # Validate model
        results[model_name] = mean_scores

    # Select the model with the best average scores
    best_model_name = max(results, key=lambda model: (results[model]['accuracy'] + results[model]['precision'] +
                                                      results[model]['recall'] + results[model]['f1']) / 4)
    best_model_score = results[best_model_name]

    st.subheader(f"Best Model: {best_model_name} with average scores:")
    st.write(
        f"Accuracy: {best_model_score['accuracy']:.4f}, Precision: {best_model_score['precision']:.4f}, Recall: {best_model_score['recall']:.4f}, F1-Score: {best_model_score['f1']:.4f}")

    return best_model_name, models[best_model_name]