import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def regression_lineaire(df, X, y, train_size, FEATURES, TARGET):
    """
    Performs linear regression on the provided data and displays the results in Streamlit.

    Args:
        df (pd.DataFrame): Original DataFrame containing the data.
        X (pd.DataFrame): DataFrame with features for regression.
        y (pd.Series): Series with the target variable.
        train_size (float): Proportion of data used for training (between 0 and 1).
        FEATURES (list): List of column names used as features.
        TARGET (str): Name of the target column.

    Returns:
        LinearRegression: Trained linear regression model.
    """
    st.write("Starting linear regression algorithm")
    random_state = 42

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = pd.DataFrame(X_test, columns=FEATURES)

    # Predict on the test set
    y_pred = model.predict(X_test)
    results_df = pd.DataFrame()
    # Create DataFrame for results
    for column in X_test.columns:
        results_df[column] = X_test[column]
    results_df['actual target'] = y_test
    results_df['predicted target'] = y_pred

    # Display results in a table in Streamlit
    st.write("Linear Regression Results:")
    st.table(results_df.head(10))

    # Visualize results with a plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title("Comparison of Actual and Predicted Targets")
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)  # Plot regression line
    st.pyplot(fig)

    return model


def regression_ridge(df, X, y, train_size=0.2):
    """
    Performs Ridge regression with different alpha parameters and displays results in Streamlit.

    Args:
        df (pd.DataFrame): Original DataFrame containing the data.
        X (pd.DataFrame): DataFrame with features for regression.
        y (pd.Series): Series with the target variable.
        train_size (float): Proportion of data used for training (between 0 and 1).

    Returns:
        Ridge: Trained Ridge regression model with the best alpha parameter.
    """
    df_resultat_ridge = []
    # Select alpha range and step using Streamlit sliders
    n_alphas = st.slider("Alpha", min_value=0.10, max_value=10.0, step=0.1)
    step = st.slider("Alpha step size", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    for alpha in alphas:
        # Train and evaluate model for each alpha
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        X_test = pd.DataFrame(X_test)
        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test.values, y_pred), 4)

        # Create DataFrame for results
        res = pd.DataFrame({"variable": X_test.columns, "coefficient": model.coef_})
        res['alpha'] = alpha
        res['mse'] = mse

        df_resultat_ridge.append(res)

    df_resultat_ridge = pd.concat(df_resultat_ridge)
    st.write("Ridge Regression Results:")
    st.write(df_resultat_ridge)
    # Find best alpha based on MSE
    alphas_used = df_resultat_ridge.groupby("alpha")['mse'].mean()
    st.write("Best MSE Score:")
    alphas_used = alphas_used.reset_index()

    # Plot MSE evolution with alpha
    st.subheader("MSE Evolution with Alpha")
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs. Alpha")
    st.pyplot(fig)

    best_alpha = alphas_used.sort_values(by='mse')['alpha'].iloc[0]
    st.success(f"Best alpha parameter for the model: {best_alpha}")
    st.line_chart(df_resultat_ridge.set_index('alpha')[['coefficient']])

    model = Ridge(alpha=best_alpha)
    return model


def regression_lasso(df, X, y, train_size=0.2):
    """
    Performs Lasso regression with different alpha parameters and displays results in Streamlit.

    Args:
        df (pd.DataFrame): Original DataFrame containing the data.
        X (pd.DataFrame): DataFrame with features for regression.
        y (pd.Series): Series with the target variable.
        train_size (float): Proportion of data used for training (between 0 and 1).

    Returns:
        Lasso: Trained Lasso regression model with the best alpha parameter.
    """
    df_resultat_lasso = []
    # Select alpha range and step using Streamlit sliders
    n_alphas = st.slider("Alpha", min_value=0.10, max_value=10.0, step=0.1)
    step = st.slider("Alpha step size", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    for alpha in alphas:
        model = Lasso(alpha=alpha).fit(X_train, y_train)
        X_test = pd.DataFrame(X_test)
        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test.values, y_pred), 4)
        res = pd.DataFrame({"variable": X_test.columns, "coefficient": model.coef_})
        res['alpha'] = alpha
        res['mse'] = mse
        df_resultat_lasso.append(res)

    df_resultat_lasso = pd.concat(df_resultat_lasso, ignore_index=True)
    st.write("Lasso Regression Results")
    st.dataframe(df_resultat_lasso)
    alphas_used = df_resultat_lasso.groupby("alpha")['mse'].mean().reset_index()

    # Plot MSE evolution with alpha
    st.subheader("MSE Evolution with Alpha")
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs. Alpha")
    st.pyplot(fig)

    best_alpha = alphas_used.sort_values(by='mse')['alpha'].iloc[0]
    st.success(f"Best alpha parameter for the model: {best_alpha}")

    model = Lasso(alpha=best_alpha)
    return model

    # st.line_chart(df_resultat_lasso.set_index('alpha')['coefficient'])


def regression_elasticnet(df, X, y, train_size=0.2):
    """
    Performs ElasticNet regression with different alpha and l1 ratio parameters.

    Args:
        df (pd.DataFrame): Original DataFrame containing the data.
        X: Independent variables.
        y: Dependent variable.
        train_size (float): Proportion of data used for training.

    Returns:
        ElasticNet: Trained ElasticNet regression model with the best alpha.
    """
    df_resultat_elasticnet = []
    # Select parameters using Streamlit sliders
    n_alphas = st.slider("Alpha", min_value=0.10, max_value=10.0, step=0.1)
    l1_ratio = st.slider("L1 Ratio", min_value=0.0, max_value=1.0, step=0.1)
    step = st.slider("Alpha step size", min_value=0.01, max_value=0.2, step=0.01)
    alphas = np.arange(0, n_alphas, step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    for alpha in alphas:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X_train, y_train)
        X_test = pd.DataFrame(X_test)
        y_pred = model.predict(X_test)
        mse = round(mean_squared_error(y_test.values, y_pred), 4)
        res = pd.DataFrame({"variable": X_test.columns, "coefficient": model.coef_})
        res['alpha'] = alpha
        res['mse'] = mse
        df_resultat_elasticnet.append(res)

    df_resultat_elasticnet = pd.concat(df_resultat_elasticnet, ignore_index=True)
    st.write("ElasticNet Regression Results")
    st.dataframe(df_resultat_elasticnet)
    alphas_used = df_resultat_elasticnet.groupby("alpha")['mse'].mean().reset_index()

    # Plot MSE evolution with alpha
    st.subheader("MSE Evolution with Alpha")
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(alphas_used['alpha'], alphas_used['mse'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs. Alpha")
    st.pyplot(fig)

    best_alpha = alphas_used.sort_values(by='mse')['alpha'].iloc[0]
    st.success(f"Best alpha parameter for the model: {best_alpha}")

    model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio)
    return model


def regression_random_forest(df, X, y, train_size=0.2):
    """
    Performs Random Forest regression on the provided data.

    Args:
        df (pd.DataFrame): Original DataFrame containing the data.
        X (pd.DataFrame): DataFrame with features for regression.
        y (pd.Series): Series with the target variable.
        train_size (float): Proportion of data used for training.

    Returns:
        RandomForestRegressor: Trained Random Forest regression model.
    """
    # Select hyperparameters using Streamlit sliders
    n_estimators = st.slider("Number of trees", min_value=10, max_value=1000, step=10)
    max_depth = st.slider("Maximum depth", min_value=1, max_value=50, step=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Train Random Forest regression model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error for Random Forest regression: {mse}")

    # Visualize results with a plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title("Comparison of Actual and Predicted Targets")
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)  # Plot regression line
    st.pyplot(fig)

    return model


def regression_gradient_boosting(df, X, y, train_size=0.2):
    """
    Performs Gradient Boosting regression on the provided data.

    Args:
        df (pd.DataFrame): Original DataFrame containing the data.
        X (pd.DataFrame): DataFrame with features for regression.
        y (pd.Series): Series with the target variable.
        train_size (float): Proportion of data used for training.

    Returns:
        GradientBoostingRegressor: Trained Gradient Boosting regression model.
    """
    # Select hyperparameters using Streamlit sliders
    n_estimators = st.slider("Number of trees", min_value=10, max_value=1000, step=10)
    learning_rate = st.slider("Learning rate", min_value=0.01, max_value=1.0, step=0.01)
    max_depth = st.slider("Maximum depth", min_value=1, max_value=50, step=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Train Gradient Boosting regression model
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                      random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error for Gradient Boosting regression: {mse}")

    # Visualize results with a plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title("Comparison of Actual and Predicted Targets")
    sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', ax=ax)  # Plot regression line
    st.pyplot(fig)

    return model













