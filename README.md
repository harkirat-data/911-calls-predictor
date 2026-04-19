911 Calls Data Analysis & Predictor

I built this project to analyze 911 emergency call data from Montgomery County, PA. 
I performed end-to-end data analysis starting from raw data exploration all the way 
to deploying a live machine learning web app on Streamlit Cloud.

Dataset
I worked with a dataset of 663,522 emergency calls recorded between 2015–2020.
The data contains the following fields:-

lat : String variable, Latitude
lng: String variable, Longitude
desc: String variable, Description of the Emergency Call
zip: String variable, Zipcode
title: String variable, Title
timeStamp: String variable, YYYY-MM-DD HH:MM:SS
twp: String variable, Township
addr: String variable, Address
e: String variable, Dummy variable (always 1)

I extracted the call reason (EMS, Fire, Traffic) from the title column.

EDA
I analyzed 663K+ 911 calls exploring distributions by reason, day, month and hour. 
I built heatmaps, time series plots per reason (EMS/Fire/Traffic), top township 
rankings, and boxplot to uncover call patterns across time and location.

Machine Learning
I trained Logistic Regression, Decision Tree and Random Forest to predict call reason 
from time/location features.
Random Forest and Decision Tree achieved ~56% accuracy.

Live App
[911 Calls Predictor](https://harkirat-data-911-calls-predictor.streamlit.app) — 
Built and deployed on Streamlit Cloud. Enter location, time and day to get 
a live prediction with confidence scores.

TechStack
Python • Pandas • Seaborn • Plotly • Scikit-learn • Streamlit
