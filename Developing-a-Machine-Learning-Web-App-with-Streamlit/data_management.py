import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data(file, separator, decimal='.'):
    """
    Loads a CSV file into a DataFrame.

    Args:
        file (str): The path to the CSV file.
        separator (str): The separator used in the CSV file (e.g., ',' or ';').
        decimal (str, optional): The decimal separator. Default is '.'.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file, sep=separator, decimal=decimal)
    return df


def data_overview(df):
    # Displays an overview of the dataset.
    st.write("Overview of the dataset:")
    st.dataframe(df)


def data_dimension(df):
    # Displays the dimensions of the dataset (number of rows and columns).
    st.write("Dataset Dimensions:")
    st.write("Rows =", df.shape[0], "\nColumns", df.shape[1])


def data_type(df):
    # Displays the data types of each column in the dataset.
    st.write("Data Types:")
    st.write(df.dtypes)


def descriptive_statistics(df):
    # Displays descriptive statistics for the dataset.
    st.write("Descriptive Statistics:")
    st.write(df.describe())


def missing_data(df):
    # Displays a matrix visualization of the missing values in the dataset.
    msno.matrix(df)
    st.pyplot()


def missing_data_stats(df):
    # Displays the number of missing values per variable in the dataset.
    st.write("Number of missing values per variable:")
    st.write(df.isnull().sum())


def displaying_outliers(df):
    # Displays the distribution of selected features in the dataset.
    selected_columns = st.multiselect('Select features to display', df.columns.tolist(), key='feature_selection')
    if st.button("Display", key='display_outliers_button'):
        if len(selected_columns) > 0:
            with st.container():
                for column in selected_columns:
                    plt.figure(figsize=(8, 4))
                    sns.histplot(df[column], kde=True)
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Frequency')
                    st.pyplot(plt.gcf())
        else:
            st.warning("Please select at least one feature.")


def target_analyse():
    pass


def correlation_matrix(df: pd.DataFrame):
    # Displays the correlation matrix and a heatmap for the dataset.
    correlation_matrix = df.corr()
    st.write("Correlation Matrix:")
    st.write(correlation_matrix)
    st.write("Heatmap of the Correlation Matrix:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()


def pair_plot_sns(df):
    # Displays a pair plot for the relationships between columns in the dataset.
    sns.pairplot(df)
    st.pyplot()


def preprocess(file=None, separator=',', decimal='.'):
    """
    Preprocesses a CSV file by loading the data, displaying an overview, data types,
    dimensions, descriptive statistics, and handling missing data.

    Args:
        file (str or None): The file path or file-like object to load. If None, uses Streamlit's file uploader.
        separator (str): The separator used in the CSV file (e.g., ',' or ';'). Default is ','.
        decimal (str): The decimal separator used in the CSV file. Default is '.'.

    Returns:
        pd.DataFrame or None: The DataFrame containing the data or None if no file is loaded.
    """
    # If no file is provided, use Streamlit to upload a file
    if file is None:
        file = st.file_uploader("Choose a file", type=["csv"])
        separator = st.selectbox("Separator", options=[",", ";"])
        decimal = st.selectbox("Decimal separator", options=[".", ","])

    if file is not None:
        df = load_data(file, separator, decimal)
        data_overview(df)
        data_type(df)
        data_dimension(df)
        descriptive_statistics(df)
        missing_data_stats(df)
        missing_data(df)
        st.subheader("Feature Distribution")
        displaying_outliers(df)
        # pair_plot_sns(df)

        return df
    return None
