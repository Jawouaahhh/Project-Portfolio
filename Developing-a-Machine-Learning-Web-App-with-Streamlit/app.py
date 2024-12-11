import machine_learning
import streamlit as st
import data_management
import preprocessing

# Streamlit page configuration
st.set_page_config(page_title="Machine Learning Project", layout="wide")

# Sidebar title
st.sidebar.title("Contents")

# List of available pages
pages = ["Load Data", "Data Management", "Machine Learning"]

# Radio button for navigation between pages
page = st.sidebar.radio("Go to page:", pages)

# Initialize data variable
data = None

# If the selected page is "Load Data"
if page == pages[0]:
    # Call the preprocess function to load and preprocess data
    data = data_management.preprocess()
    # Save the preprocessed data in the session state
    st.session_state["result"] = data

# If the selected page is "Data Management"
elif page == pages[1]:
    # Check if data has been loaded
    if "result" in st.session_state:
        # Call the run function from the preprocessing module to manage data
        data = preprocessing.run(st.session_state["result"])
        # Save the processed data in the session state
        st.session_state["result"] = data

# If the selected page is "Machine Learning"
elif page == pages[2]:
    # Check if data has been loaded and preprocessed
    if "result" in st.session_state:
        # Call the run function from the machine_learning module to execute machine learning models
        machine_learning.run(st.session_state["result"])
