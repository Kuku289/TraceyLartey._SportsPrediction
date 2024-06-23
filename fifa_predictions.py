import streamlit as st
import pickle
from streamlit_option_menu import option_menu

# Setting page configuration
st.set_page_config(page_title="FIFA Predictions", layout="wide", page_icon="⚽")

# Loading the saved models
nb_model = pickle.load(
    open("./trained_models/best_trained_naive_bayes_model.pkl", "rb")
)
lr_model = pickle.load(
    open("./trained_models/best_trained_linear_regression_model.pkl", "rb")
)
try:
    knn_model = pickle.load(open("./trained_models/best_trained_knn_model.pkl", "rb"))
    rf_model = pickle.load(
        open("./trained_models/best_trained_random_forest_model.pkl", "rb")
    )
except Exception as e:
    st.error(f"Error loading Random Forest and KNN models: {e}")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Multiple ML Models for FIFA Prediction System",
        ["Naïve Bayes", "Random Forest", "Linear Regression", "K-Nearest Neighbor"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0,
    )


# Function to convert user input to float and handle errors
def convert_to_float(inputs):
    try:
        result = [float(input_) for input_ in inputs]
        return result
    except ValueError:
        st.error(f"Invalid input. Please enter a numeric value.")
        return None


# Function to create the input fields and collect user input
def get_user_input():
    col1, col2 = st.columns(2)
    with col1:
        value_eur = st.text_input("Market Value in Euros")
    with col2:
        wage_eur = st.text_input("Wage in Euros")
    with col1:
        passing = st.text_input("Passing")
    with col2:
        dribbling = st.text_input("Dribbling")
    with col1:
        movement_reactions = st.text_input("Movement Reactions")
    with col2:
        mentality_composure = st.text_input("Mentality Composure")

    user_input = [
        value_eur,
        wage_eur,
        passing,
        dribbling,
        movement_reactions,
        mentality_composure,
    ]

    # If any of the inputs are None (invalid), return None to indicate failure
    if any(input_ is None for input_ in user_input):
        return None

    return user_input


# Naïve Bayes Prediction Page
if selected == "Naïve Bayes":
    # Page title
    st.title("FIFA Prediction Using ML Naïve Bayes")

    # Getting the input data from the user
    user_input = get_user_input()

    # Creating a button for Prediction
    if st.button("Test Result of Player's Overall Rating Using Naïve Bayes"):
        user_input = convert_to_float(user_input)
        if user_input:
            fifa_nb_overall_prediction = nb_model.predict([user_input])
            st.success(f"Predicted Overall Rating: {fifa_nb_overall_prediction[0]}")
        else:
            st.error("Please provide valid inputs for all fields.")

# Random Forest Prediction Page
if selected == "Random Forest":
    # Page title
    st.title("FIFA Prediction Using ML Random Forest")

    # Getting the input data from the user
    user_input = get_user_input()

    # Creating a button for Prediction
    if st.button("Test Result of Player's Overall Rating Using Random Forest"):
        user_input = convert_to_float(user_input)
        if user_input:
            fifa_rf_overall_prediction = rf_model.predict([user_input])
            st.success(f"Predicted Overall Rating: {fifa_rf_overall_prediction[0]}")
        else:
            st.error("Please provide valid inputs for all fields.")

# Linear Regression Prediction Page
if selected == "Linear Regression":
    # Page title
    st.title("FIFA Prediction Using ML Linear Regression")

    # Getting the input data from the user
    user_input = get_user_input()

    # Creating a button for Prediction
    if st.button("Test Result of Player's Overall Rating Using Linear Regression"):
        user_input = convert_to_float(user_input)
        if user_input:
            fifa_lr_overall_prediction = lr_model.predict([user_input])
            st.success(f"Predicted Overall Rating: {fifa_lr_overall_prediction[0]}")
        else:
            st.error("Please provide valid inputs for all fields.")

# K-Nearest Neighbor Prediction Page
if selected == "K-Nearest Neighbor":
    # Page title
    st.title("FIFA Prediction Using ML K-Nearest Neighbor")

    # Getting the input data from the user
    user_input = get_user_input()

    # Creating a button for Prediction
    if st.button("Test Result of Player's Overall Rating Using K-Nearest Neighbor"):
        user_input = convert_to_float(user_input)
        if user_input:
            fifa_knn_overall_prediction = knn_model.predict([user_input])
            st.success(f"Predicted Overall Rating: {fifa_knn_overall_prediction[0]}")
        else:
            st.error("Please provide valid inputs for all fields.")
