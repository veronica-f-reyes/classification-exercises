 
# Functions to obtain Telco Customer data from the Codeup Data Science Database: telco_churn
#It returns a pandas dataframe.
#--------------------------------

#This function uses my info from my env file to create a connection url to access the Codeup db.  

from typing import Container
import pandas as pd
import os
from env import host, user, password

#Function to connect to database for SQL query use
def get_db_url(host, user, password, database):
        
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
    return url


#Function to get data from telco_churn database
def get_telco_data():
    
    filename = "telco_churn.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:

        database = 'telco_churn'

        #Create SQL query to select data from telco_churn database
        query = '''
                SELECT * FROM customers;
                '''

         # read the SQL query into a dataframe
        df = pd.read_sql(query, get_db_url(host,user, password, database))

         # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

#Call function to get and create telco_churn.csv locally
get_telco_data()





