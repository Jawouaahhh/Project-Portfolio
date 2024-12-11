import streamlit as st
import pandas as pd
from data_management import data_type, missing_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)


def change_variable_types(df):
    """
    Changes the data type of a selected column in a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    Returns:
        pd.DataFrame: The updated DataFrame with the new data type.
    """
    column_name = st.selectbox("Select your variable:", options=df.columns.to_list())
    new_type = st.selectbox("Select the new type:", ["float", "int", "object", "string"])
    if st.button("Convert"):
        df[column_name] = df[column_name].astype(new_type)
        data_type(df)
    return df


def delete_columns(df):
    """
    Deletes selected columns from a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    Returns:
        pd.DataFrame: The updated DataFrame after deleting columns.
    """
    columns_to_delete = df.columns.to_list()
    columns_name = st.multiselect("Select column(s) to delete:", columns_to_delete)
    if st.button("Delete"):
        for column_name in columns_name:
            df.drop(column_name, axis=1, inplace=True)
        st.success('Columns deleted')
        st.dataframe(df)
    return df


def columns_with_missing_data(df):
    # Returns the names of columns with missing values.
    return df.columns[df.isnull().any()]


def get_list_strategy(df, column_to_treat):
    # Returns a list of treatment strategies for a given column.
    if df[column_to_treat].dtypes in ['float64', 'int64']:
        return ["effacer", "moyenne", "médiane", "valeur personnalisée"]
    else:
        return ["effacer", "valeur personnalisée"]


def missed_data_treatment(df):
    """
    Handles missing data in a DataFrame using various strategies.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    Returns:
        pd.DataFrame: The DataFrame with missing values treated.
    """
    columns_na = columns_with_missing_data(df)
    if columns_na is not None and len(columns_na) > 0:
        column_to_treat = st.selectbox("Select column to treat", columns_na)
        list_strategy = get_list_strategy(df, column_to_treat)
        fill_strategy = st.selectbox("Select strategy to handle missing values", list_strategy)

        if fill_strategy == "valeur personnalisée":
            custom_value = st.text_input("Enter custom value")
            if st.button("Fill Missing Values with Custom Value"):
                df[column_to_treat] = df[column_to_treat].fillna(custom_value)
                st.success("Replacement complete")
        elif fill_strategy == "effacer":
            if st.button("Delete", key="button_Delete"):
                df.dropna(subset=column_to_treat, inplace=True)
                st.success("Deletion complete")
        else:
            if st.button(f"Fill Missing Values with {fill_strategy}"):
                df[column_to_treat] = df[column_to_treat].fillna(
                    df[column_to_treat].mean() if fill_strategy == "mean" else df[column_to_treat].median())
                st.success("Replacement complete")
    else:
        st.write("Your dataset is complete.")
    st.subheader("Cleaned Data Visualization")
    st.dataframe(df)
    missing_data(df)
    return df


def matrice_correlation(numerical_df):
    # Displays the correlation matrix as a heatmap.
    correlation_matrix = numerical_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot()


def pair_plot(numerical_df):
    # Generates pair plots for numerical variables in the DataFrame.
    sns.pairplot(numerical_df)
    st.pyplot()


def plot_data(df):
    # Plots the correlation matrix and pair plot for selected numerical columns.
    numerical_df = df.select_dtypes(include=['number'])
    columns_corr = st.multiselect("Select columns to visualize", numerical_df.columns, key='corr_columns')
    if st.button("Display Correlations", key='display_corr_button'):
        if len(columns_corr) > 0:
            numerical_df = numerical_df[columns_corr]
            with st.container():
                matrice_correlation(numerical_df)
                pair_plot(numerical_df)
        else:
            st.warning("Please select at least one column.")


def run(df):
    """
    Executes a series of data processing steps on a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    Returns:
        pd.DataFrame: The DataFrame after processing.
    """
    if df is not None:
        st.subheader("Change Variable Types")
        df = change_variable_types(df)
        st.subheader("Handle Empty Columns")
        df = delete_columns(df)
        st.subheader("Handle Missing Values")
        df = missed_data_treatment(df)
        st.subheader("Correlation Matrix")
        plot_data(df.head())
        st.subheader("Dataset Size")
        st.write(df.shape)
        return df
