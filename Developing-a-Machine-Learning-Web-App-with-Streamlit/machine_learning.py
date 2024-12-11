import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from regression import regression_lineaire, regression_ridge, regression_lasso, regression_elasticnet, \
    regression_random_forest, regression_gradient_boosting
from classification import tree_classifier, k_neighbors_classifier, logistic_regression, support_vector_classifier, \
    naive_bayes_classifier, xgboost_classifier, sgd_classifier
from validation import validation_k_fold_classification, validation_k_fold_regression, compare_regression_models, \
    compare_classification_models

# Define available algorithm types and models
ML_ALGO_TYPES = ["Regression", "Classification"]
REGRESSION_MODELS = ["Linear Regression", "Lasso", "Ridge", "ElasticNet", "Random Forest", "Gradient Boosting"]
CLASSIFICATION_MODELS = ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbors Classifier",
                         "Support Vector Classifier", "Naive Bayes Classifier", "XGBoost Classifier",
                         "Stochastic Gradient Descent Classifier"]


def select_algo_type():
    """
    Allow the user to select the type of machine learning algorithm (Regression or Classification).

    Returns:
        str: Selected algorithm type.
    """
    ml_algo_type = st.selectbox("Choose the type of learning algorithm", options=ML_ALGO_TYPES)
    return ml_algo_type


def select_algo_model(algo_type):
    """
    Allow the user to select a specific machine learning model based on the algorithm type.

    Args:
        algo_type (str): Type of algorithm selected (Regression or Classification).

    Returns:
        str: Selected model.
    """
    models = []
    if algo_type == ML_ALGO_TYPES[0]:
        models = REGRESSION_MODELS
    elif algo_type == ML_ALGO_TYPES[1]:
        models = CLASSIFICATION_MODELS
    model = st.selectbox("Select the learning model", options=models)
    return model


def is_standardized(X):
    """
    Check if the dataset is already standardized.

    Args:
        X (pd.DataFrame): Input data.

    Returns:
        bool: True if the dataset is standardized, otherwise False.
    """
    return X.apply(lambda x: x.mean()).abs().lt(0.01).all() and X.apply(lambda x: x.std()).subtract(1).abs().lt(
        0.01).all()


def data_standardisation(X):
    """
    Standardize the input features if they are not already standardized.

    Args:
        X (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Standardized input data.
    """
    if X is not None and is_standardized(X):
        st.write("Your dataset is already standardized")
    else:
        if st.checkbox("Standardize your data"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
    return X


def target_encoding(y):
    """
    Encode the target variable if it is not numeric.

    Args:
        y (pd.Series): Target variable.

    Returns:
        pd.Series: Encoded target variable.
    """
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        label_mapping_df = pd.DataFrame(
            {'Category': label_encoder.classes_, 'Encoded': label_encoder.transform(label_encoder.classes_)})
        st.subheader("Target Encoding")
        st.write(pd.DataFrame(label_mapping_df))
    return y


def feature_encoding(X):
    """
    Encode non-numeric features in the dataset.

    Args:
        X (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Encoded input data.
    """
    if X is not None and len(X) > 0:
        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                label_encoder = LabelEncoder()
                X[column] = label_encoder.fit_transform(X[column])
        st.dataframe(X)
    return X


def select_features(df):
    """
    Allow the user to select features for the model.

    Args:
        df (pd.DataFrame): Dataframe containing all potential features.

    Returns:
        list: List of selected features.
    """
    features = st.multiselect("Choose your explanatory variables (FEATURES)", df.columns.to_list())
    return features


def select_target(df):
    """
    Allow the user to select the target variable.

    Args:
        df (pd.DataFrame): Dataframe containing all potential targets.

    Returns:
        str: Selected target variable.
    """
    target = st.selectbox("Choose your target variable (TARGET)", df.columns.to_list())
    return target


def run_machine_learning(df, ml_algo_type, ml_algo_model, X, y, FEATURES, TARGET):
    """
    Execute machine learning based on the provided parameters.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        ml_algo_type (str): The type of algorithm (regression or classification).
        ml_algo_model (str): The specific model algorithm.
        X (pd.DataFrame): Training features.
        y (pd.Series): Target variable.
        FEATURES (list): List of feature column names.
        TARGET (str): Target variable column name.

    Returns:
        None
    """
    model = None
    if ml_algo_type == ML_ALGO_TYPES[0]:
        train_size = st.slider("Test set size", min_value=0.0, max_value=1.0, step=0.05, value=0.2)
        if ml_algo_model == REGRESSION_MODELS[0]:
            model = regression_lineaire(df, X, y, train_size, FEATURES, TARGET)
        if ml_algo_model == REGRESSION_MODELS[1]:
            model = regression_lasso(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[2]:
            model = regression_ridge(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[3]:
            model = regression_elasticnet(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[4]:
            model = regression_random_forest(df, X, y, train_size)
        if ml_algo_model == REGRESSION_MODELS[5]:
            model = regression_gradient_boosting(df, X, y, train_size)
    elif ml_algo_type == ML_ALGO_TYPES[1]:
        if ml_algo_model == CLASSIFICATION_MODELS[0]:
            valeur = st.number_input("Enter test_size value:", step=0.01, value=0.02)
            valeur = 0.2
            model = logistic_regression(X, y, valeur)
        if ml_algo_model == CLASSIFICATION_MODELS[1]:
            model = tree_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[2]:
            model = k_neighbors_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[3]:
            model = support_vector_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[4]:
            model = naive_bayes_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[5]:
            model = xgboost_classifier(X, y)
        if ml_algo_model == CLASSIFICATION_MODELS[6]:
            model = sgd_classifier(X, y)
    if model is not None:
        st.subheader("Cross Validation")
        if ml_algo_type == ML_ALGO_TYPES[0]:
            validation_k_fold_regression(X, y, FEATURES, model)
        elif ml_algo_type == ML_ALGO_TYPES[1]:
            validation_k_fold_classification(X, y, FEATURES, model)


def run(df):
    """
    Run the machine learning workflow in the Streamlit application.

    Args:
        df (pd.DataFrame): The dataframe containing the data.

    Returns:
        pd.DataFrame: Updated dataframe.
    """
    st.subheader("Select Target Variable")
    # Select target variable
    TARGET = select_target(df)
    y = df[TARGET]

    if y is not None:
        st.subheader("Target Column")
        st.dataframe(y.head())

    # Encode the target variable
    y = target_encoding(y)

    st.subheader("Select Features")
    # Select features
    FEATURES = select_features(df)
    X = df[FEATURES]

    # Encode the features
    X = feature_encoding(X)

    # Standardize the features
    X = data_standardisation(X)

    st.subheader("Select Algorithm Type (Regression / Classification)")
    # Select algorithm type
    ml_algo_type = select_algo_type()

    st.subheader("Select Learning Model")
    # Select learning model
    ml_algo_model = select_algo_model(ml_algo_type)

    st.dataframe(X)
    st.subheader("Start Learning")
    if st.checkbox("Start the process"):
        run_machine_learning(df, ml_algo_type, ml_algo_model, X, y, FEATURES, TARGET)
    if st.checkbox("Start Comparison"):
        if ml_algo_type == ML_ALGO_TYPES[0]:
            compare_regression_models(X, y, FEATURES)
        elif ml_algo_type == ML_ALGO_TYPES[1]:
            compare_classification_models(X, y, FEATURES)
    return df
