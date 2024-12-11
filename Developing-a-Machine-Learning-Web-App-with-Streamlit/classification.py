import streamlit as st
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def plot_confusion_matrix(y_test, y_pred, figsize=(6, 3)):
    """
    Display a confusion matrix using a heatmap.

    Parameters:
    y_test : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels by the model.
    figsize : tuple, optional, default (6, 3)
        Size of the figure for the confusion matrix plot.

    This function uses Streamlit to display the plot in a web application.
    """
    # Display subtitle for the confusion matrix
    st.subheader("Confusion Matrix")

    # Create a new figure with the specified size
    plt.figure(figsize=figsize)

    # Calculate the confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    # Create a heatmap for the confusion matrix
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Reds", cbar=False)

    # Label the axes
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Display the plot in Streamlit
    st.pyplot()


def tree_classifier(X, y, test_size=0.2):
    """
    Train and evaluate a decision tree classifier.

    Parameters:
    X : DataFrame
        Features used for training.
    y : Series or array-like
        Class labels.
    test_size : float, optional, default 0.2
        Proportion of the dataset to use as the test set.

    This function uses Streamlit to display results in a web application.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create and train the decision tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate model accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy in Streamlit
    st.success(f"Accuracy: {accuracy:.2f}")

    # Generate and display the classification report in Streamlit
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Classification Report")
    st.table(classification_report_dict)

    # Display the decision tree graphically in Streamlit
    st.subheader("Decision Tree Visualization")
    plt.figure(figsize=(15, 10))
    X = pd.DataFrame(X)
    plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(i) for i in model.classes_])
    st.pyplot()

    # Display the confusion matrix using Seaborn
    plot_confusion_matrix(y_test, y_pred)
    return model


def logistic_regression(X, y, test_size=0.2):
    """
    Train and evaluate a logistic regression model.

    Parameters:
    X : DataFrame
        Features used for training.
    y : Series or array-like
        Class labels.
    test_size : float, optional, default 0.2
        Proportion of the dataset to use as the test set.

    This function uses Streamlit to display results in a web application.
    """
    # Create the logistic regression model
    model = LogisticRegression()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the class labels on the test set
    y_pred = model.predict(X_test)

    # Evaluate the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy in Streamlit
    st.success(f"Accuracy: {accuracy:.2f}")

    # Probabilities of test dataset
    y_prob = model.predict_proba(X_test)

    # Display the classification report in Streamlit
    st.subheader('Classification Report')
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
    st.table(classification_report_dict)

    # Display the confusion matrix in Streamlit
    plot_confusion_matrix(y_test, y_pred)

    return model


def k_neighbors_classifier(X, y, test_size=0.2):
    """
    Train and evaluate a K-Nearest Neighbors (KNN) classifier.

    Parameters:
    X : DataFrame
        Features used for training.
    y : Series or array-like
        Class labels.
    test_size : float, optional, default 0.2
        Proportion of the dataset to use as the test set.

    This function uses Streamlit to display results in a web application.
    """
    # Display the model title in Streamlit
    st.write("K-Nearest Neighbors Classifier")

    # Create the K-Nearest Neighbors model with 3 neighbors
    model = KNeighborsClassifier(n_neighbors=3)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the class labels on the test set
    y_pred = model.predict(X_test)

    # Evaluate the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy in Streamlit
    st.success(f"Accuracy: {accuracy:.2f}")

    # Display the classification report in Streamlit
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(report)

    # Display the confusion matrix in Streamlit
    plot_confusion_matrix(y_test, y_pred)

    return model


def support_vector_classifier(X, y, test_size=0.2):
    """
    Train and evaluate a Support Vector Classifier (SVC).

    Parameters:
    X : DataFrame
        Features used for training.
    y : Series or array-like
        Class labels.
    test_size : float, optional, default 0.2
        Proportion of the dataset to use as the test set.

    Returns:
    model : SVC
        The trained SVC model.

    This function uses Streamlit to display results in a web application.
    """
    # Display the model title in Streamlit
    st.write("Support Vector Classifier")

    # Create the Support Vector Classifier model with probability enabled
    model = SVC(probability=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the class labels on the test set
    y_pred = model.predict(X_test)

    # Evaluate the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy in Streamlit
    st.success(f"Accuracy: {accuracy:.2f}")

    # Display the classification report in Streamlit
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Classification Report")
    st.table(report)

    # Display the confusion matrix in Streamlit
    plot_confusion_matrix(y_test, y_pred)

    return model


def naive_bayes_classifier(X, y, test_size=0.2):
    """
    Train and evaluate a Naive Bayes Classifier.

    Parameters:
    X : DataFrame
        Features used for training.
    y : Series or array-like
        Class labels.
    test_size : float, optional, default 0.2
        Proportion of the dataset to use as the test set.

    Returns:
    model : GaussianNB
        The trained Naive Bayes model.

    This function uses Streamlit to display results in a web application.
    """
    # Display the model title in Streamlit
    st.write("Naive Bayes Classifier")

    # Create the Naive Bayes model
    model = GaussianNB()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the class labels on the test set
    y_pred = model.predict(X_test)

    # Evaluate the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy in Streamlit
    st.success(f"Accuracy: {accuracy:.2f}")

    # Display the classification report in Streamlit
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Classification Report")
    st.table(report)

    # Display the confusion matrix in Streamlit
    plot_confusion_matrix(y_test, y_pred)

    return model


def xgboost_classifier(X, y, test_size=0.2):
    """
    Train and evaluate an XGBoost Classifier.

    Parameters:
    X : DataFrame
        Features used for training.
    y : Series or array-like
        Class labels.
    test_size : float, optional, default 0.2
        Proportion of the dataset to use as the test set.

    Returns:
    model : XGBClassifier
        The trained XGBoost model.

    This function uses Streamlit to display results in a web application.
    """
    # Display the model title in Streamlit
    st.write("XGBoost Classifier")

    # Create the XGBoost model with a fixed random state for reproducibility
    model = XGBClassifier(random_state=42)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the class labels on the test set
    y_pred = model.predict(X_test)

    # Evaluate the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy in Streamlit
    st.success(f"Accuracy: {accuracy:.2f}")

    # Generate the classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Display the classification report in Streamlit
    st.subheader("Classification Report")
    st.table(report)

    # Display the confusion matrix in Streamlit
    plot_confusion_matrix(y_test, y_pred)

    return model


def sgd_classifier(X, y, test_size=0.2):
    """
    Train and evaluate a Stochastic Gradient Descent (SGD) Classifier.

    Parameters:
    X : DataFrame
        Features used for training.
    y : Series or array-like
        Class labels.
    test_size : float, optional, default 0.2
        Proportion of the dataset to use as the test set.

    Returns:
    model : SGDClassifier
        The trained SGD model.

    This function uses Streamlit to display results in a web application.
    """
    # Display the model title in Streamlit
    st.write("Stochastic Gradient Descent Classifier")

    # Create the SGD model with a fixed random state for reproducibility
    model = SGDClassifier(random_state=42)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the class labels on the test set
    y_pred = model.predict(X_test)

    # Evaluate the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy in Streamlit
    st.success(f"Accuracy: {accuracy:.2f}")

    # Generate the classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Display the classification report in Streamlit
    st.subheader("Classification Report")
    st.table(report)

    # Display the confusion matrix in Streamlit
    plot_confusion_matrix(y_test, y_pred)

    return model
