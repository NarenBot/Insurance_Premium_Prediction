import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


from src.logger import logging
from src.exception import CustomException
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.database import DatabaseConnect


st.set_page_config(page_title="Insurance::Home")


# CONFIG.yml
with open(r"config/auth_config.yml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)

authentication_status = authenticator.login("Login...", "main")

# LOGIN AUTHENTICATION
if st.session_state["authentication_status"] == False:
    st.error("Username/Password is incorrect.")

elif st.session_state["authentication_status"] == None:
    st.warning("Please enter your username and password.")

elif st.session_state["authentication_status"] == True:
    # LOGOUT BUTTON FOR ALL PAGES
    logout = authenticator.logout("Logout", "main", key="unique_key")
    title = st.title("*Insurance_Premium_Prediction*")
    tag = st.markdown(
        "##### (--Approach To Forecast Health Insurance Premiums For Individuals.)"
    )

    # FILE UPLOADER
    # st.help(st.form)   # Help function.
    st.sidebar.file_uploader("Choose the file.")
    # page = st.sidebar.selectbox(
    #     "Pages", ["About", "Training Pipeline", "Prediction Pipeline", "Show Dataframe"]
    # )
    page = st.sidebar.radio(
        "Select the options below for your convenience.",
        [
            "About",
            "Training Pipeline",
            "Prediction Pipeline",
            "Show Dataframe",
            "Contact",
        ],
    )

    # ABOUT PAGE
    if page == "About":
        st.subheader("About:")
        st.markdown(
            "The purposes of this exercise to look into different features to observe their relationship, and plot a multiple linear regression based on several features of individual such as age, physical/family condition and location against their existing medical expense to be used for predicting future medical expenses of individuals that help medical insurance to make decision on charging the premium."
        )

    # TRAINING PIPELINE PAGE
    if page == "Training Pipeline":
        st.markdown("***")
        training = st.button("Do you need to train the past data? Click Here!")
        if training:
            st.info("Please wait for sometime, the training is going on...")
            train = TrainPipeline()
            tuned_score = train.run_pipeline()
            accuracy = str(round(tuned_score, 2) * 100)[:2]
            st.success(f"Model trained successfully with accuracy: {accuracy}%")

    # PREDICTION PIPELINE PAGE
    if page == "Prediction Pipeline":
        st.markdown("***")
        st.info("All fields are mandatory!")

        # FORM COMPONENTS
        form_submit = False
        with st.form(key="my_form", clear_on_submit=True):
            name = st.text_input(
                "Please enter your name.", placeholder="Your name please..."
            )
            age = st.number_input("Please enter your age.", 0, 100)
            sex = st.selectbox(
                "Please enter your sex.",
                ["male", "female"],
                index=None,
                placeholder="Choose...",
            )
            bmi = st.number_input("Please enter your bmi.", 0, 50)
            children = st.number_input("How many children?", 0, 10)
            smoker = st.selectbox(
                "Are you a smoker?", ["yes", "no"], index=None, placeholder="Choose..."
            )
            region = st.selectbox(
                "Please enter your region.",
                ["northeast", "northwest", "southeast", "southwest"],
                index=None,
                placeholder="Choose...",
            )
            prediction = st.form_submit_button("Submit")
            form_submit = True if prediction else False

        if form_submit:
            input_data = CustomData(age, sex, bmi, children, smoker, region)
            dataframe = input_data.get_data_as_dataframe()
            preds = PredictPipeline()
            results = preds.prediction(dataframe)
            st.success("Predicted Insurance Amount ₹ {:.2f}".format(float(results[0])))

            # DATABASE CONNECTION
            connect = DatabaseConnect()
            insert_data = connect.insert_user_data(
                name,
                age,
                sex,
                bmi,
                children,
                smoker,
                region,
                round(float(results[0]), 2),
            )
            st.info(
                "To view our database, kindly navigate to the 'Show Dataframe' page."
            )

    # SHOW DATAFRAME PAGE
    if page == "Show Dataframe":
        st.markdown("***")
        st.info("*** User Details ***")
        connect = DatabaseConnect()
        data, cursor = connect.display_user_database()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        st.dataframe(df.tail())

    if page == "Contact":
        st.markdown("***")
        st.subheader("*--Contact Details--*")
        st.text("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "[![Narendran Mudadi|LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/narendran-mudadi/)"
            )
        with col2:
            st.markdown(
                "[![Narendran Mudadi|G-Mail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:narendas10@gmail.com?subject=Insurance_Premium)"
            )
