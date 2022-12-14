import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

from pandas.io.json import json_normalize


def GetGlobalAircraftData():
    # General Aircraft DataSet. This contains a globabl DB of aircraft of several types. For the purpose of my analysis
    # I am only interested in large commercial aircraft so this will need to be cleaned up to suit my needs.
    # Source: https://www.kaggle.com/datasets/ahmedeltom/open-air-traffic-data-opensky-netwrok?select=aircraft-database-complete-2022-11-cleaned.csv

    # load full aircraft DB
    df = pd.read_csv("aircraft-database-complete-2022-11-cleaned.csv")

    # Show raw data
    print(df.info())  # commenting out - used for dev purposes [will clean up later]
    print(df.describe())
    ShowMissingValues('BEFORE', 'AIRCRAFT', df)  # function to show missing values before and after data cleaning

    # Cleaning the data for my usage
    # -------------------------------
    # Fix the column name and set to ID for the first column
    df.rename(columns={df.columns[0]: 'ID'}, inplace=True)
    df.set_index(['ID', 'Manufacturer', 'Aircraft_family'])

    # I only care about large commercial aircraft in my dataset so filtering out for Manufacturer of Airbus, Boeing & ATR
    # Also, manufacturer is sparcely filled out however I can use company instead if manufacturer is blank,
    # before filtering on manufacturer I need to fill na with Company.
    df['Manufacturer'].fillna(df['Company'], inplace=True)

    missing = df[df['Manufacturer'].isnull() & df['Aircraft_type'].str.contains('Boeing')]
    # print(missing.head(20)) # commenting out - used for dev purposes

    # If Manufacturer is still null then check and derive the Manufacturer from Aircraft type
    df['Manufacturer'] = np.where(df['Aircraft_type'].str.contains('ATR') & df['Manufacturer'].isnull(), 'ATR', df['Manufacturer'])
    df['Manufacturer'] = np.where(df['Aircraft_type'].str.contains('Airbus') & df['Manufacturer'].isnull(), 'AIRBUS', df['Manufacturer'])
    df['Manufacturer'] = np.where(df['Aircraft_type'].str.contains('Boeing') & df['Manufacturer'].isnull(), 'BOEING', df['Manufacturer'])

    # Check for missing core fields? commenting out now as only using for analysis purposes when developing
    # missing = df[df['Registration'].isnull()] #change to check different columns
    # print(missing.head(20))

    # Now filter the dataset to only show the core manufacturers but convert to upper first just in case
    df['Manufacturer'] = df['Manufacturer'].str.upper()
    df = df.loc[df['Manufacturer'].isin(['AIRBUS', 'BOEING', 'ATR'])]

    # Now drop columns we probably wont use
    df = df.drop(['Line_number', 'Classification', 'Emitter', 'Company'], axis=1)

    # I still have 25 missing registration and most also do not have a serial number so these are usless records for my analysis so dropping
    missing = df[df['Registration'].isnull()]
    # print(missing.info()) # commenting out - used during dev
    df = df.dropna(subset=['Registration'])

    # Check if we have any duplicates - both work and show no duplicates
    print(df.duplicated(keep='last'))
    print(df.loc[df.duplicated(), :])

    ShowMissingValues('AFTER', 'AIRCRAFT', df)  # function to show missing values before and after data cleaning
    print(df.describe())

    return df


def GetFlightsData():
    # load flights data from 2015. This dataset is old however I wanted a dataset of flight details and delay
    # information that contained the aircraft tail number for future analysis
    # Source dataset - https://www.kaggle.com/code/pierrekos/analysis-and-prediction-of-aircraft-delays/data?select=flights.csv

    # This a large dataset to load each time during development so cuting down the size to make it quicker to run the code.
    # ################## DON'T FORGET TO CHANGE THIS LATER ##################################
    # ---------------Swap comment lines below for first and last run ----------------------------------
    flights = pd.read_csv("flights-2015.csv")
    # flights = pd.read_csv("flights-2015-shortfile.csv")

    # As per above - Using for quicker runs while developing - create small dataset with 500 rows - Only run once
    # flights = flights.iloc[:5000,:]
    # flights.to_csv("flights-2015-shortfile.csv")

    # Show raw data
    print(flights.info())  # commenting out - used during dev
    print(flights.describe())
    ShowMissingValues('BEFORE', 'FLIGHTS', flights)  # function to show missing values before and after data cleaning

    # Creating new column to convert separate date time fields into single column and create a day column
    flights['DEPARTURE_DATE'] = pd.to_datetime(flights.YEAR * 10000 + flights.MONTH * 100 + flights.DAY, format='%Y%m%d')

    flights['DAY_OF_WEEK'] = flights['DEPARTURE_DATE'].dt.day_name()

    # Create a new column called BLOCK_TIME to compare against AIR_TIME.
    # This column need to include taxi in and out time. We want to get the total time we have the engines running
    flights['BLOCK_TIME'] = flights.AIR_TIME + flights.TAXI_OUT + flights.TAXI_IN
    flights['BLOCK_FLIGHT_VARIANCE'] = flights.BLOCK_TIME - flights.AIR_TIME

    # Fix the column name and set to ID for the first column
    flights.rename(columns={flights.columns[0]: 'ID'}, inplace=True)
    flights.set_index(['ID'])

    # Add status level for delay levels
    for dataset in flights:
        flights.loc[flights['ARRIVAL_DELAY'] <= 15, 'DELAY_STATUS'] = 0
        flights.loc[flights['ARRIVAL_DELAY'] >= 15, 'DELAY_STATUS'] = 1
        flights.loc[flights['ARRIVAL_DELAY'] >= 60, 'DELAY_STATUS'] = 2
        flights.loc[flights['DIVERTED'] == 1, 'DELAY_STATUS'] = 3
        flights.loc[flights['CANCELLED'] == 1, 'DELAY_STATUS'] = 4

    # Change cancellations cause to something more manageable
    flights.loc[flights["CANCELLATION_REASON"] == "A", 'CANCELLATION_REASON'] = "0"
    flights.loc[flights["CANCELLATION_REASON"] == "B", 'CANCELLATION_REASON'] = "1"
    flights.loc[flights["CANCELLATION_REASON"] == "C", 'CANCELLATION_REASON'] = "2"
    flights.loc[flights["CANCELLATION_REASON"] == "D", 'CANCELLATION_REASON'] = "3"

    # dropping the unwanted data
    # flights = flights.drop('YEAR', 1)  # Converted to date
    # flights = flights.drop("MONTH", 1)  # Converted to date
    # flights = flights.drop("DAY", 1)  # Converted to date
    # flights = flights.drop("DAY_OF_WEEK", 1)
    flights = flights.drop("FLIGHT_NUMBER", 1)
    flights = flights.drop("SCHEDULED_DEPARTURE", 1)
    flights = flights.drop("ELAPSED_TIME", 1)
    flights = flights.drop("SCHEDULED_ARRIVAL", 1)
    # Note: these may be used for later analysis but geting rid for now (Note2: adding back in for delay analysis)
    # flights = flights.drop("DIVERTED", 1)
    # flights = flights.drop("CANCELLED", 1)
    # flights = flights.drop("CANCELLATION_REASON", 1)
    # flights = flights.drop("AIR_SYSTEM_DELAY", 1)
    # flights = flights.drop("SECURITY_DELAY", 1)
    # flights = flights.drop("AIRLINE_DELAY", 1)
    # flights = flights.drop("LATE_AIRCRAFT_DELAY", 1)
    # flights = flights.drop("WEATHER_DELAY", 1)

    # print(flights.head(12)) #commenting out - used for dev

    ShowMissingValues('AFTER', 'FLIGHTS', flights)  # function to show missing values before and after data cleaning
    print(flights.info())  # commenting out - used during dev
    print(flights.describe())

    return flights


# Generic function to show missing values in a dataframe before and after data cleaning
def ShowMissingValues(when, datasetname, dataframe):
    # TODO: Remove this line at the end. Saving time while in dev
    # return

    missing = dataframe.isnull().sum().div(dataframe.shape[0]).sort_values(ascending=False)
    plt.figure(figsize=(9, 9))
    title = 'Missing values (%) - ' + datasetname + ' - (' + when + ')'
    plt.title(title, fontsize=16, fontweight='bold')
    sns.barplot(x=missing.values, y=missing.index, palette='Blues_r')
    plt.xlim(0, 1)
    plt.show()

    return


def GetAirportsListFromAPI():
    # airports
    request = requests.get('http://api.aviationstack.com/v1/airports?access_key=636cee5fb0b4a0b713e5c16e2f595e3e')
    data = request.json()
    # for p in data['data']:
    #    print(p['airport_name'])

    # res = json_normalize(data)
    # print(res)

    # return as dataframe - commented out - used in testing
    # return pd.DataFrame(res)
    # return json
    return data


def GetAirlinesListFromAPI():
    # airlines
    request = requests.get('http://api.aviationstack.com/v1/airlines?access_key=636cee5fb0b4a0b713e5c16e2f595e3e')
    data = request.json()
    # for p in data['data']:
    #    print(p['airline_name'] + ' - ' + p['icao_code'])

    # res = json_normalize(data)
    # print(res)

    # return as dataframe - commented out - used in testing
    # return pd.DataFrame(res)
    # return json
    return data