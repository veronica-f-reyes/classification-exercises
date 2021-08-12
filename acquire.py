#Make a function named get_titanic_data that returns the titanic data 
# from the codeup data science database as a pandas data frame. 
# Obtain your data from the Codeup Data Science Database.

def get_titanic_data():
    from env import host, user, password

   #def get_db_url(host, user, password, database):
        
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
    #return url

    query = "SELECT * FROM passengers;"

    df = pd.read_sql(query, get_db_url(host,user, password, 'titanic_db'))

    return df


#Make a function named get_iris_data that returns the data from the iris_db 
# on the codeup data science database as a pandas data frame. The returned 
# data frame should include the actual name of the species
# in addition to the species_ids. Obtain your data from the Codeup Data Science Database.

#def get_iris_data():



#Once you've got your get_titanic_data and get_iris_data functions written, 
# now it's time to add caching to them. To do this, edit the beginning of
# the function to check for a local filename like titanic.csv or iris.csv. 
# If they exist, use the .csv file. If the file doesn't exist, then produce 
# the SQL and pandas necessary to create a dataframe, then write the dataframe
# to a .csv file with the appropriate name.

#Be sure to add titanic.csv and iris.csv to your .gitignore file