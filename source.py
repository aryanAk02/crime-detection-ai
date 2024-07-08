import yaml
from streamlit_authenticator import Authenticate
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(fields='main')

if authentication_status:
    st.sidebar.success(f'Welcome **{name}**')

    logo = Image.open('logo.jpg')
    st.sidebar.image(logo)

    st.sidebar.error('This project introduces a comprehensive system that leverages machine learning techniques to address critical aspects of law enforcement—crime prediction and policing quality assessment. ')

    st.sidebar.markdown('---')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    from statsmodels.tsa.arima.model import ARIMA

    st.title("CRIME DETECTION")

    dataset_choice = st.sidebar.selectbox('SELECT TYPE OF CRIME',
                                          options=['PROPERTY STOLEN & RECOVERED',
                                                   'COMPLAINTS AGAINST POLICE'])

    if dataset_choice == 'PROPERTY STOLEN & RECOVERED':

        data = pd.read_csv('Property_stolen_and_recovered.csv')

        area_name_list = data['area_name'].unique().tolist()
        area_name_list_index = np.arange(1, len(area_name_list) + 1, 1)
        area_name_text_to_num = {i: j for i, j in zip(area_name_list, area_name_list_index)}
        area_name_num_to_text = {j: i for i, j in zip(area_name_list, area_name_list_index)}

        group_name_list = data['group_name'].unique().tolist()
        group_name_list_index = np.arange(1, len(group_name_list) + 1, 1)
        group_name_text_to_num = {i: j for i, j in zip(group_name_list, group_name_list_index)}
        group_name_num_to_text = {j: i for i, j in zip(group_name_list, group_name_list_index)}

        sub_group_name_list = data['sub_group_name'].unique().tolist()
        sub_group_name_list_index = np.arange(1, len(sub_group_name_list) + 1, 1)
        sub_group_name_text_to_num = {i: j for i, j in zip(sub_group_name_list, sub_group_name_list_index)}
        sub_group_name_num_to_text = {j: i for i, j in zip(sub_group_name_list, sub_group_name_list_index)}

        data['area_name'] = data['area_name'].map(area_name_text_to_num)
        data['group_name'] = data['group_name'].map(group_name_text_to_num)
        data['sub_group_name'] = data['sub_group_name'].map(sub_group_name_text_to_num)

        data_sorted = data.sort_values(by=['area_name', 'group_name', 'sub_group_name', 'year'])

        data_sorted['cases_recovery_rate'] = data_sorted['cases_property_recovered'] / data['cases_property_stolen']
        data_sorted.fillna(1, inplace=True)

        data_sorted['property_value_recovery_rate'] = data_sorted['value_of_property_recovered'] / data[
            'value_of_property_stolen']
        data_sorted.fillna(1, inplace=True)
        mean_property_recovery_rate = data_sorted['cases_property_stolen'].mean()

        area_name_code_input = st.sidebar.selectbox('enter area name : ', options=area_name_text_to_num)
        group_name_code_input = st.sidebar.selectbox('enter group name : ', options=group_name_text_to_num)

        property_theft_submit_button = st.sidebar.button('SUBMIT')

        if property_theft_submit_button:
            import matplotlib.pyplot as plt

            data_clustered = pd.read_csv('property_stolen_and_recovered_clustered.csv')

            area_name_text = area_name_text_to_num[area_name_code_input]
            group_name_text = group_name_text_to_num[group_name_code_input]

            filter1 = (data_clustered['area_name'] == area_name_text) & (
                    data_clustered['group_name'] == group_name_text)
            data_filtered = data_clustered[filter1]

            st.subheader("MODEL DIAGNOSIS")
            cluster_1_img = Image.open('cluster_1.png')
            st.image(cluster_1_img)

            st.error(f"The case solving rate for **{area_name_code_input}** - **{group_name_code_input}** is: **{data_filtered['case_recovery_class'].mode().iloc[0]}**")

            cluster_2_img = Image.open('cluster_2.png')
            st.image(cluster_2_img)

            st.warning(f"The property recovery rate for **{area_name_code_input}** - **{group_name_code_input}** is: **{data_filtered['property_recovery_class'].mode().iloc[0]}**")


            st.subheader('FILTERED DATA FOR GIVEN PARAMETERS')
            data_filtered

            st.markdown('---')

            # CASE RECOVERY RATE ARIMA
            # Perform ARIMA forecasting for cases_recovery_rate
            y_train_value_recovery_rate = data_filtered['cases_recovery_rate']

            # Fit ARIMA model
            order = (1, 1, 1)  # You may need to adjust the order based on your data characteristics
            model = ARIMA(y_train_value_recovery_rate, order=order)
            fit_model = model.fit()

            next_year = data_filtered['year'].iloc[-1] + 1

            # Forecast for the next 5 years
            future_years = np.arange(next_year, next_year + 5, 1)
            future_predictions = fit_model.get_forecast(steps=5).predicted_mean

            # Plot the actual data, forecast, and connect with a red line
            plt.figure(figsize=(8, 6))
            plt.plot(data_filtered['year'], data_filtered['cases_recovery_rate'], label='Actual', marker='o')
            plt.plot(future_years, future_predictions.values, marker='o', color='red', label='Predicted')

            # Connect the points with a red line
            plt.plot(data_filtered['year'].iloc[-1:], data_filtered['cases_recovery_rate'].iloc[-1:], color='red')
            plt.plot([data_filtered['year'].iloc[-1], future_years[0]],
                     [data_filtered['cases_recovery_rate'].iloc[-1], future_predictions.values[0]], color='red')

            st.subheader(f'Case Recovery Rate for {area_name_code_input} with 5-Year Forecast')
            plt.xlabel('Year')
            plt.ylabel('Cases Recovery Rate')
            plt.legend()
            plt.grid(True)
            st.pyplot()

            # Display model evaluation parameters
            st.subheader('MODEL PARAMETERS FOR CASE RECOVERY RATE')
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            st.success(
                f"Mean Absolute Error : {mean_absolute_error(y_train_value_recovery_rate, fit_model.fittedvalues)}")
            st.error(
                f"Mean Squared Error : {mean_squared_error(y_train_value_recovery_rate, fit_model.fittedvalues)}")
            st.info(
                f"Root Mean Squared Error : {np.sqrt(mean_squared_error(y_train_value_recovery_rate, fit_model.fittedvalues))}")



            st.markdown('---')
            # PROPERTY VALUE RATE ARIMA
            # Perform ARIMA forecasting for cases_recovery_rate
            y_train = data_filtered['property_value_recovery_rate']

            # Fit ARIMA model
            order = (1, 1, 1)  # You may need to adjust the order based on your data characteristics
            model = ARIMA(y_train, order=order)
            fit_model_2 = model.fit()

            next_year = data_filtered['year'].iloc[-1] + 1

            # Forecast for the next 5 years
            future_years = np.arange(next_year, next_year + 5, 1)
            future_predictions = fit_model_2.get_forecast(steps=5).predicted_mean

            # Plot the actual data, forecast, and connect with a red line
            plt.figure(figsize=(8, 6))
            plt.plot(data_filtered['year'], data_filtered['property_value_recovery_rate'], label='Actual', marker='o')
            plt.plot(future_years, future_predictions.values, marker='o', color='red', label='Predicted')

            # Connect the points with a red line
            plt.plot(data_filtered['year'].iloc[-1:], data_filtered['property_value_recovery_rate'].iloc[-1:],
                     color='red')
            plt.plot([data_filtered['year'].iloc[-1], future_years[0]],
                     [data_filtered['cases_recovery_rate'].iloc[-1], future_predictions.values[0]], color='red')

            st.subheader(f'Property Value Recovery Rate for {area_name_code_input} with 5-Year Forecast')
            plt.xlabel('Year')
            plt.ylabel('Property Value Recovery Rate')
            plt.legend()
            plt.grid(True)
            st.pyplot()

            # Display model evaluation parameters
            st.subheader('MODEL PARAMETERS FOR PROPERTY VALUE RECOVERY RATE')
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            st.success(
                f"Mean Absolute Error : {mean_absolute_error(y_train, fit_model_2.fittedvalues)}")
            st.error(
                f"Mean Squared Error : {mean_squared_error(y_train, fit_model_2.fittedvalues)}")
            st.info(
                f"Root Mean Squared Error : {np.sqrt(mean_squared_error(y_train, fit_model_2.fittedvalues))}")


    if dataset_choice == 'COMPLAINTS AGAINST POLICE':
        import matplotlib.pyplot as plt

        df = pd.read_csv('Complaints_against_police.csv')
        selected_columns = ['Area_Name', 'Year', 'CPA_-_Cases_Registered',
                            'CPA_-_Complaints_Received/Alleged',
                            'CPB_-_Police_Personnel_Convicted',
                            'CPC_-_Police_Personnel_Dismissal/Removal_from_Service',
                            'CPC_-_Police_Personnel_Major_Punishment_awarded']

        df = df[selected_columns]
        # Rename columns
        df.columns = ['area_name', 'year',
                      'cases_registered', 'complaints_received',
                      'police_convicted', 'police_removed_from_service', 'police_punished']

        df['total_punishment_cases'] = df['police_convicted'] + df['police_removed_from_service'] + df[
            'police_punished']

        # Calculate ratios
        df['received_registered_ratio'] = df['cases_registered'] / df['complaints_received']
        df['registered_police_punished_ratio'] = df['police_punished'] / df['cases_registered']

        area_name_list = df['area_name'].unique().tolist()
        area_name_list_index = np.arange(1, len(area_name_list) + 1, 1)
        area_name_text_to_num = {i: j for i, j in zip(area_name_list, area_name_list_index)}
        area_name_num_to_text = {j: i for i, j in zip(area_name_list, area_name_list_index)}

        # df['area_name'] = df['area_name'].map(area_name_text_to_num)
        df_sorted = df.sort_values(by=['area_name', 'year'])
        complaint_against_police_area = st.sidebar.selectbox('choose a state',
                                                             options=area_name_text_to_num)

        complaint_against_police_submit_button = st.sidebar.button('submit')

        if complaint_against_police_submit_button:
            data_clustered = pd.read_csv('complaints_against_police_clustered.csv')
            filter1 = (data_clustered['area_name'] == complaint_against_police_area)
            data_clustered_filtered = data_clustered[filter1]

            st.subheader("MODEL DIAGNOSIS")
            cluster_3_img = Image.open('cluster_3.png')
            st.image(cluster_3_img)

            st.error(f"The complaint registration rate for **{complaint_against_police_area}** is: **{data_clustered_filtered['complaint_registration_class'].mode().iloc[0]}**")

            cluster_4_img = Image.open('cluster_4.png')
            st.image(cluster_4_img)

            st.warning(f"The police punishment rate for **{complaint_against_police_area}** is: **{data_clustered_filtered['police_punished_class'].mode().iloc[0]}**")

            filter2 = (df_sorted['area_name'] == complaint_against_police_area)
            df_filtered = df_sorted[filter2]

            st.subheader("FILTERED DATA FOR COMPLAINTS AGAINST POLICE")
            df_filtered

            # CASES REGISTERED vs CASES RECEIVED RATIO ARIMA
            # Perform ARIMA forecasting for cases_recovery_rate
            y_train_case_registration_rate = df_filtered['received_registered_ratio']

            # Fit ARIMA model
            order = (1, 1, 1)  # You may need to adjust the order based on your df characteristics
            model = ARIMA(y_train_case_registration_rate, order=order)
            fit_model_crr = model.fit()

            next_year = df_filtered['year'].iloc[-1] + 1

            # Forecast for the next 5 years
            future_years = np.arange(next_year, next_year + 5, 1)
            future_predictions = fit_model_crr.get_forecast(steps=5).predicted_mean

            st.markdown('---')

            # Plot the actual df, forecast, and connect with a red line
            plt.figure(figsize=(8, 6))
            plt.plot(df_filtered['year'], df_filtered['received_registered_ratio'], label='Actual', marker='o')
            plt.plot(future_years, future_predictions.values, marker='o', color='red', label='Predicted')

            # Connect the points with a red line
            plt.plot(df_filtered['year'].iloc[-1:], df_filtered['received_registered_ratio'].iloc[-1:], color='red')
            plt.plot([df_filtered['year'].iloc[-1], future_years[0]],
                     [df_filtered['received_registered_ratio'].iloc[-1], future_predictions.values[0]], color='red')

            st.subheader(f'Complaint Registration Rate for {complaint_against_police_area} with 5-Year Forecast')
            plt.xlabel('Year')
            plt.ylabel('Complaint Registration Rate')
            plt.legend()
            plt.grid(True)
            st.pyplot()

            # Display model evaluation parameters
            st.subheader('MODEL PARAMETERS FOR CASE REGISTRATION RATE')
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            st.success(
                f"Mean Absolute Error : {mean_absolute_error(y_train_case_registration_rate, fit_model_crr.fittedvalues)}")
            st.error(
                f"Mean Squared Error : {mean_squared_error(y_train_case_registration_rate, fit_model_crr.fittedvalues)}")
            st.info(
                f"Root Mean Squared Error : {np.sqrt(mean_squared_error(y_train_case_registration_rate, fit_model_crr.fittedvalues))}")


            st.markdown('---')
            # POLICE PUNISHED vs CASES RECEIVED RATIO ARIMA
            # Perform ARIMA forecasting for cases_recovery_rate
            y_train_police_punished_rate = df_filtered['registered_police_punished_ratio']

            # Fit ARIMA model
            order = (1, 1, 1)  # You may need to adjust the order based on your df characteristics
            model = ARIMA(y_train_case_registration_rate, order=order)
            fit_model_ppr = model.fit()

            next_year = df_filtered['year'].iloc[-1] + 1

            # Forecast for the next 5 years
            future_years = np.arange(next_year, next_year + 5, 1)
            future_predictions = fit_model_ppr.get_forecast(steps=5).predicted_mean

            # Plot the actual df, forecast, and connect with a red line
            plt.figure(figsize=(8, 6))
            plt.plot(df_filtered['year'], df_filtered['registered_police_punished_ratio'], label='Actual', marker='o')
            plt.plot(future_years, future_predictions.values, marker='o', color='red', label='Predicted')

            # Connect the points with a red line
            plt.plot(df_filtered['year'].iloc[-1:], df_filtered['registered_police_punished_ratio'].iloc[-1:],
                     color='red')
            plt.plot([df_filtered['year'].iloc[-1], future_years[0]],
                     [df_filtered['registered_police_punished_ratio'].iloc[-1], future_predictions.values[0]],
                     color='red')

            st.subheader(f'Police Punishment Rate for {complaint_against_police_area} with 5-Year Forecast')
            plt.xlabel('Year')
            plt.ylabel('Police Punishment Rate')
            plt.legend()
            plt.grid(True)
            st.pyplot()

            # Display model evaluation parameters
            st.subheader('MODEL PARAMETERS FOR POLICE PUNISHED RATE')
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            st.success(
                f"Mean Absolute Error : {mean_absolute_error(y_train_police_punished_rate, fit_model_ppr.fittedvalues)}")
            st.error(
                f"Mean Squared Error : {mean_squared_error(y_train_police_punished_rate, fit_model_ppr.fittedvalues)}")
            st.info(
                f"Root Mean Squared Error : {np.sqrt(mean_squared_error(y_train_police_punished_rate, fit_model_ppr.fittedvalues))}")


    st.sidebar.markdown('---')

    authenticator.logout('Log out','sidebar')

elif authentication_status == False:
    st.error('Username/password is incorrect')

elif authentication_status == None:
    st.warning('Please enter your username and password')


