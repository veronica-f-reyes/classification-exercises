# Make a function named get_titanic_data that returns the titanic data 
# from the codeup data science database as a pandas data frame. 
# Obtain your data from the Codeup Data Science Database.

#This function uses my info from my env file to create a connection url to access the Codeup db.  

import pandas as pd
import os
from env import host, user, password

#Function to connect to database for SQL query use
def get_db_url(host, user, password, database):
        
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
    return url


#Function to get data from Titanic database
def get_titanic_data():
    
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:

        database = 'titanic_db'

        #Create SQL query to select data from Titanic database
        query = "SELECT * FROM passengers;"

         # read the SQL query into a dataframe
        df = pd.read_sql(query, get_db_url(host,user, password, 'titanic_db'))

         # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

#Call function to get and create titanic.csv locally
get_titanic_data()

#Make a function named get_iris_data that returns the data from the iris_db 
# on the codeup data science database as a pandas data frame. The returned 
# data frame should include the actual name of the species
# in addition to the species_ids. Obtain your data from the Codeup Data Science Database.

def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
    
        database = 'iris_db'

        query = "SELECT * FROM species;"

        # read the SQL query into a dataframe
        df = pd.read_sql(query, get_db_url(host,user, password, 'iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

#Call function to get and create iris.csv locally
get_iris_data()



