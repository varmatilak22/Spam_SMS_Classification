import psycopg2
import pandas as pd
from sqlalchemy import create_engine 

#Database credentials
db_username='postgres'
db_password='1234'
db_host='localhost'
db_name='sampledb'
db_port='5432'

#Read the CSV file
data=pd.read_csv('spam.csv',encoding='ISO-8859-1')
print(data)

#Get the columns names
print(data.columns)

#Create an SQLALCHEMY ENGINE
engine=create_engine(f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}")



#Write a pandas dataframe to the database
data.to_sql('spam',engine,index=False,if_exists='replace')

print("Data Loaded Successfully")