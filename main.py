import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import random

#======================================================================================================================
#
# Assessment UCDPA_DAMIEN_CORMICAN
#
# Date: 14 Dec 2022
#
# Core Modules
# main.py - core python file to call the function to load and clean datasets and also create visualizations and insights
# LoadandCleanDatasets.py - python file to load and clean the various datasets
# OracleDatabaseAccess.py - sed during the development process to connect to my corporate database to retrieve fleet details
#
#======================================================================================================================


#change display window for pycharm output to see more than 3 columns
pd.options.display.width = 0

# ===============================
# LOAD AND CLEAN DATASETS SECTION
# ===============================

# Get and load my core datasets. Note: cleaning of dataset also complete in these functions
#-----------------------------------------------------
from LoadandCleanDatasets import GetGlobalAircraftData, GetFlightsData, GetAirlinesListFromAPI, GetAirportsListFromAPI

# 2 core datasets
aircraft = GetGlobalAircraftData()   # full global list of aircraft - see function header for more info
flights = GetFlightsData()           # detailed list of flight data (5.8 million flights) - see function header for more info

#API feed for lookup data
airlines = GetAirlinesListFromAPI()
airports = GetAirportsListFromAPI()


#print(airlines.info())
#print(airlines.head(15))
#print(airports.info())
#print(airports.head(15))


# ================================
# MERGE DATASETS AND FINAL TIDY UP
# ================================

# Next task is to join the datasets with an inner join on tail number/registration. Note, this is the
# same data point although with different name on the separate datasets

df = pd.merge(flights, aircraft,  how='inner', left_on = 'TAIL_NUMBER', right_on = 'Registration')
print(df.head(150))

# Due to the nature of the datasets and also the timing of the datasets (Global fleet is current but flight
# data is US based only and from 2015) we will have flight data with no corresponding data in the global aircraft list.
# This is OK for assessment purposes and the final target is to replace the flights dataset with a realtime API to retrieve global
# flight data from the previous 3 months and extract my fleet from an internal corporate Oracle DB. Obviously I can't use this data
# for this course work but I have connected to the Oracle DB to retrieve my corporate fleet. I am leaving the code
# in this project file but have removed the connection into input requests - see OracleDatabaseAccess for more info.

#Review list of Airlines
#print(df['Callsign'].unique())     # Commenting out for dev purposes only

missing = df[df['Aircraft_type'].isnull()]
print(missing.head(10))
#if we still have missing Aircraft Type at this stage then remove
df = df.dropna(subset=['Aircraft_type'])
print(df.head(15))

unique = df['Registration'].unique()
print(unique.size)

print(df.info())  # commenting out - used for dev purposes [will clean up later]
print(df.describe())

# Check for missing core fields? commenting out now as only using for analysis purposes when developing
missing = df[df['Registration'].isnull()]  #change to check different columns
print(missing.head(20))

exit()

# ====================
# ANALYSE DATA SECTION
# ====================

# ========================================
# 1. INITIAL HIGH LEVEL REVIEW OF DATASETS
# ========================================

#Correlation Matrix - I saw this used in another area of Kaggle and thought it could be useful to identify data coralation
corrmat = flights.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.title("Correlation Matrix - Flight Data", fontsize=14)
plt.ylabel("Flight Data Fields")
plt.xlabel("Flight Data Fields")
plt.show()


# pie plot to show flights stats by the days of week
f,ax=plt.subplots(1,2,figsize=(14,6))
df['DAY_OF_WEEK'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Day Of Week')
ax[0].set_ylabel('')
sns.countplot(x=df['DAY_OF_WEEK'], data=df, ax=ax[1])
ax[1].set_title('Day Of Week')
ax[1].set_ylabel('')
ax[1].set_xlabel('')
plt.show()

# pie plot to show flights stats by month
f,ax=plt.subplots(1,2,figsize=(14,6))
df['MONTH'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Month')
ax[0].set_ylabel('')
sns.countplot(x=df['MONTH'], data=df, ax=ax[1])
ax[1].set_title('Month')
ax[1].set_ylabel('')
ax[1].set_xlabel('')
plt.show()


# Complete some high level analysis on the global list of aircraft
grouped = aircraft[['Manufacturer','ID']].groupby(['Manufacturer']).count()
print(grouped)
grouped.plot(kind="pie", figsize=(10, 9), subplots=True)
plt.show()

grouped = aircraft[['Aircraft_family','ID']].groupby(['Aircraft_family']).count()
#grouped = grouped.nlargest(25,grouped['Aircraft_family'].value_counts())  #largest 25
#grouped = grouped.groupby(['Aircraft_family']).size().sort_values(ascending=False)
grouped = grouped.groupby('Aircraft_family').first()
print(grouped)
grouped.plot(kind="barh", figsize=(10, 9), subplots=True)
plt.show()


# TODO: show some visualizations on each dataset individually for demonstration purposes

print(flights.info())
print(flights.head(15))



# =================================
# 2. ANALYSE BLOCK TIME VS AIR TIME
# =================================

# NOTE: BLOCK_TIME AND BLOCK_FLIGHT_VARIANCE ARE ADDED CALCULATED FIELDS DURING THE DATA LOAD AND CLEAN FUNCTION

# Block to Air time mean variance by Aircraft Type
# Bar Chart with 25 highest Block to Air time variance by aircraft types
grouped = df[['Aircraft_type','BLOCK_FLIGHT_VARIANCE']].groupby(['Aircraft_type']).mean() #groupby with mean
grouped = grouped.nlargest(25,['BLOCK_FLIGHT_VARIANCE'])  #largest 25
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE', 'Aircraft_type'],ascending=True) #sort order
grouped.plot(kind="barh", figsize=(14, 9))    # set horizontal bar chart and size
#plt.legend(['Block to Flight time Variance'], loc='upper right')
plt.legend('', frameon=False)  # hide legend
plt.title("25 Highest Block to Air time variance by Aircraft Type", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Aircraft Type")
plt.show()

# Bar Chart with 25 lowest Block to Air time variance by aircraft types
grouped = df[['Aircraft_type','BLOCK_FLIGHT_VARIANCE']].groupby(['Aircraft_type']).mean()
grouped = grouped.nsmallest(25,['BLOCK_FLIGHT_VARIANCE'])
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE', 'Aircraft_type'], ascending=False)
grouped.plot(kind="barh", figsize=(10, 9))
plt.legend('', frameon=False)  # hide legend
plt.title("25 Lowest Block to Air time variance by Aircraft Type", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Aircraft Type")
plt.show()

# Block to Air time mean variance by Aircraft Family
# Bar Chart with 25 highest Block to Air time variance by aircraft Family
grouped = df[['Aircraft_family','BLOCK_FLIGHT_VARIANCE']].groupby(['Aircraft_family']).mean() #groupby with mean'
grouped = grouped.nlargest(25,['BLOCK_FLIGHT_VARIANCE'])  #largest 25
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE', 'Aircraft_family'],ascending=True) #sort order
grouped.plot(kind="barh", figsize=(14, 9))    # set horizontal bar chart and size
#plt.legend(['Block to Flight time Variance'], loc='upper right')
plt.legend('', frameon=False)  # hide legend
plt.title("25 Highest Block to Air time variance by Aircraft Family", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Aircraft Family")
plt.show()

# Bar Chart with 25 lowest Block to Air time variance by aircraft Family
grouped = df[['Aircraft_family','BLOCK_FLIGHT_VARIANCE']].groupby(['Aircraft_family']).mean()
grouped = grouped.nsmallest(25,['BLOCK_FLIGHT_VARIANCE'])
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE', 'Aircraft_family'], ascending=False)
grouped.plot(kind="barh", figsize=(10, 9))
plt.legend('', frameon=False)  # hide legend
plt.title("25 Lowest Block to Air time variance by Aircraft Family", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Aircraft Family")
plt.show()

# Block to Air time mean variance by Airline
# Bar Chart with 25 highest Block to Air time variance by Airline
grouped = df[['AIRLINE','BLOCK_FLIGHT_VARIANCE']].groupby(['AIRLINE']).mean()
grouped = grouped.nlargest(25,['BLOCK_FLIGHT_VARIANCE'])
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE', 'AIRLINE'],ascending=True)
grouped.plot(kind="barh", figsize=(10, 9))
plt.legend('', frameon=False)  # hide legend
plt.title("25 Highest Block to Air time variance by Airline", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Airline")
plt.show()

# Bar Chart with 25 lowest Block to Air time variance by Airline
grouped = df[['AIRLINE','BLOCK_FLIGHT_VARIANCE']].groupby(['AIRLINE']).mean()
grouped = grouped.nsmallest(25,['BLOCK_FLIGHT_VARIANCE'])
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE', 'AIRLINE'], ascending=False)
grouped.plot(kind="barh", figsize=(10, 9))
plt.legend('', frameon=False)  # hide legend
plt.title("25 Lowest Block to Air time variance by Airline", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Airline")
plt.show()


# Block to Air time mean variance by Route (origin airport)
# Bar Chart with 25 highest Block to Air time variance by Route (origin airport)
grouped = df[['ORIGIN_AIRPORT','BLOCK_FLIGHT_VARIANCE']].groupby(['ORIGIN_AIRPORT']).mean()
grouped = grouped.nlargest(25,['BLOCK_FLIGHT_VARIANCE'])
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE','ORIGIN_AIRPORT'],ascending=True)
grouped.plot(kind="barh", figsize=(10, 9))
plt.legend('', frameon=False)  # hide legend
plt.title("25 Highest Block to Air time variance by Route (Origin Airport)", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Route (Origin Airport)")
plt.show()

# Bar Chart with 25 lowest Block to Air time variance by Route (origin airport)
grouped = df[['ORIGIN_AIRPORT','BLOCK_FLIGHT_VARIANCE']].groupby(['ORIGIN_AIRPORT']).mean()
grouped = grouped.nsmallest(25,['BLOCK_FLIGHT_VARIANCE'])
grouped = grouped.sort_values(by=['BLOCK_FLIGHT_VARIANCE','ORIGIN_AIRPORT'], ascending=False)
grouped.plot(kind="barh", figsize=(10, 9))
plt.legend('', frameon=False)  # hide legend
plt.title("25 Lowest Block to Air time variance by Route (Origin Airport)", fontsize=14)
plt.ylabel("Variance of Block to Flight Time (minutes)")
plt.xlabel("Route (Origin Airport)")
plt.show()


# ===================================
# 3. FLIGHT HOURS PER MONTH BY TAIL
# ===================================

# The section is in progress and will be utilised fully after the assessment as hitting a corporate Oracle DB.
# For the purpose of the assessment I am generating a random list of ID's and using this to extract a 'simulated fleet' of aircraft
#

# Generate 150 random numbers between 1 and 2000
randomlist = random.sample(range(10, 2000), 25)
print(randomlist)

# Using the random list to generate a fleet of aircraft up to 150 and use this to select against ID_x
# This will be replaced by my company's actual fleet after the assessment
myfleet = df.loc[df['ID_x'].isin(randomlist)]
print(myfleet)

# Not using as part of the assessment but used in testing. This is hitting a corporate DB so obviously can't include here
#from OracleDatabaseAccess import getoracledataset
#myfleet = getoracledataset('***add fleetview here***')


monthly_hours = myfleet[['TAIL_NUMBER', 'MONTH', 'AIR_TIME']].groupby(['TAIL_NUMBER', 'MONTH']).sum()
print(monthly_hours)
monthly_hours.plot(kind="line", figsize=(14, 9), )    # set horizontal bar chart and size
#######plt.legend(['Block to Flight time Variance'], loc='upper right')
plt.legend('', frameon=False)  # hide legend
plt.title("My Fleet - Monthly Aircraft Utilisation", fontsize=14)
plt.ylabel("Flight Time (minutes)")
plt.xlabel("Aircraft")
plt.show()




# ==========================
# 4. ANALYSE FLIGHT DELAY STATS
# ==========================


# Display visual of delayed flight
delayed_subset = df
f,ax=plt.subplots(1,2,figsize=(14,8))
df['DELAY_STATUS'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Delay Status')
ax[0].set_ylabel('')
sns.countplot(x=df['DELAY_STATUS'],order = df['DELAY_STATUS'].value_counts().index, data=df, ax=ax[1])
ax[1].set_title('Delay Status')
ax[0].legend(ncol = 2, loc = 'lower right')
ax[1].legend(ncol = 2, loc = 'lower right')
plt.show()

print('Status represents wether the flight was on time (0), slightly delayed (1), highly delayed (2), diverted (3), or cancelled (4)')

# Display a visual of cancelled flights
cancelled_flights = df[(df.DELAY_STATUS == 4)]
#print(cancelled_flights.info())
f,ax=plt.subplots(1,2,figsize=(14,8))
cancelled_flights['CANCELLATION_REASON'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_ylabel('')
ax[0].legend(ncol = 2, loc = 'upper right')
ax[1].legend(ncol = 2, loc = 'upper right')
sns.countplot(x=df['CANCELLATION_REASON'], order = cancelled_flights['CANCELLATION_REASON'].value_counts().index, data=cancelled_flights, ax=ax[1])
plt.show()
#print('0 = carrier, 1 = weather, 2 = NAS')

cancelled_flights[['DEPARTURE_DATE','CANCELLATION_REASON']].groupby(['DEPARTURE_DATE']).count().plot()
plt.show()


# Delayed Flights

delayed_flights = df[(df.DELAY_STATUS >= 1) &(df.DELAY_STATUS < 3)]

#Hide this chart
#sns.distplot(x=delayed_flights['ARRIVAL_DELAY'])
#plt.show()

f,ax=plt.subplots(1,2,figsize=(14,8))
delayed_flights[['MONTH','ARRIVAL_DELAY']].groupby(['MONTH']).mean().plot(ax=ax[0])
ax[0].set_title('Average delay by month')
delayed_flights[['MONTH','ARRIVAL_DELAY']].groupby(['MONTH']).sum().plot(ax=ax[1])
ax[1].set_title('Number of minutes delayed by month')
ax[0].legend(ncol = 2, loc = 'upper right')
ax[1].legend(ncol = 2, loc = 'upper right')
plt.show()

#sns.jointplot(x='DEPARTURE_TIME',y='ARRIVAL_DELAY',data=delayed_flights,kind='reg', color='b',fit_reg = True)
#plt.show()

df2 = delayed_flights.filter(['MONTH','AIRLINE_DELAY','WEATHER_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY'], axis=1)
df2 = df2.groupby('MONTH')['LATE_AIRCRAFT_DELAY','AIRLINE_DELAY','WEATHER_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY'].sum().plot()
df2.legend(loc='upper right', shadow=True)
plt.show()

#scatterplot - removing this. It is nice but not sure what it is telling me and it is slow
#sns.set()
#cols = ['ARRIVAL_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'WEATHER_DELAY']
#sns.pairplot(delayed_flights[cols], size = 2.5)
#plt.show()

