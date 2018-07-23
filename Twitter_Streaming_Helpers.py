import numpy as np
import pandas as pd
import tweepy
import json
import subprocess
import datetime
import MySQLdb
import sqlalchemy
import time

def Get_Twitter_Data_As_Pandas(C_Key, C_Secret, A_Token, A_Token_Secret, Max_Tweets= 100, Filters = ["Trump"], Table_Name = "tmp_tweets", New_Table_Columns = None, Query = None, Tweet_Data_Parts = None, Temporary = True, *args, **kwargs):
    """
   
    C_Key,
        String - Twitter Consumer Key
    C_Secret, 
        String - Twitter Consumer Secret
    A_Token,
        String - Twitter Access Token
    A_Token_Secret, 
        String - Twitter Access Token Secret
    Max_Tweets= 100,
        INT - Number of tweets to extract
    Filters = None, 
        List(String) - What words to filter on
        Default - ["Trump"]
    Table_Name = None,
        String - The name of your new table (default is tmp_tweets)
    New_Table_Columns = "(date DATETIME, username VARCHAR(20), tweet VARCHAR(280))",
        List(String) - SQL format tuple of string pairs for column name and type e.g. ['time DATETIME', 'age INT(2)']'
    Query = None,
        String - SQL query to execute in database table
    Tweet_Data_Parts = None
        List(String/List(String)) - Parts of the tweet json (according to twitter) to extract e.g. [{"user":"screen_name"}, text'] is default
        Time is automatically added in to database
    Temporary = True,
        Bool - Store Tweets in the Database temporarily or permanently
        Default = True
    
    """
    
    if not Query:
        Query = sqlalchemy.select([sqlalchemy.Text(Table_Name)])
    else:
        pass


    auth = tweepy.OAuthHandler(consumer_key=C_Key, consumer_secret=C_Secret)
    auth.set_access_token(A_Token, A_Token_Secret)

    db_connection = MySQLdb.connect("localhost","root", "Pass", "tweet_store", charset = 'utf8mb4')
    cursor = db_connection.cursor()

    tweet_add_milestone = int(Max_Tweets/5)
    
    # ## Define a class to listen to the twitter API
    # If we want to use twitter data and/or a database other than the default then define this custom listener:
    if Tweet_Data_Parts and New_Table_Columns:    
        class Stream_Listener(tweepy.StreamListener):
            def __init__(self, api=None, Max_Tweets_=None, Table_Name_=None, New_Table_Columns_=None, Tweet_Data_Parts_ = None):
                super().__init__()
                self.num_tweets = 0
                self.max_tweets = Max_Tweets_        
                self.table_name = Table_Name_
                self.tweet_data_parts = Tweet_Data_Parts_
                self.new_table_columns = New_Table_Columns_
                
                # For creating, create table if not default
                # Below line  is hide your warning 
                cursor.execute("SET sql_notes = 0; ")

                # create table here....
            
                exec_stmt = str("CREATE TABLE IF NOT EXISTS " + self.table_name + " " + self.new_table_columns) 
                
                cursor.execute(exec_stmt)
                db_connection.commit()

            def on_data(self, data):
                        if self.num_tweets < self.max_tweets:
                            all_data = json.loads(data) 
                            data_parts = ()
                           
                            for part in self.tweet_data_parts:
                                if isinstance(part, str):
                                    if part == "created_at":
                                        time_created = datetime.datetime.strptime("%a %b %d %H:%M:%S %z %Y")
                                        data_parts += time_created
                                    else:  
                                        data_parts += (all_data[part],)
                                    
                                elif isinstance(part, dict):
                                    data_parts += (all_data[part.key()][part.item()],) 
                                
                                else:
                                    raise ValueError("The Listed Tweet_Data_Part was not either of type dict or str")

                            num_inserted_vars = len(data_parts)
                            
                            exec_stmt = str("INSERT INTO " + self.table_name, "("+str(", ".join(self.new_table_columns))+")", "VALUES", str(("%s, " for i in range(num_inserted_vars))))

                            cursor.execute(exec_stmt, data_parts)

                            db_connection.commit()

                            if self.num_tweets%tweet_add_milestone == 0:    
                                print("Successfully added tweet. Number:", self.num_tweets +1)
                            self.num_tweets +=1

                            return True

                        else:
                            print("Finished writing to table:", self.table_name)
                            return False

            def on_error(self, status):
                print("Error Code:", status)

    #If we haven't proper;y defined how to create and insert into the database then raise an error
    elif (not Tweet_Data_Parts) != (not New_Table_Columns):
            raise ValueError("Need both New Table Columns and Tweet_Data_Parts to specify alternative tweet data collection")

    #Otherwise use the default listener and database/table
    else:
        class Stream_Listener(tweepy.StreamListener):
            def __init__(self, api=None, Max_Tweets_=None, Table_Name_=None, New_Table_Columns_=None, Tweet_Data_Parts_ = None):
                super().__init__()
                self.num_tweets = 0
                self.max_tweets = Max_Tweets_        
                self.table_name = Table_Name_
                self.tweet_data_parts = Tweet_Data_Parts_
                self.new_table_columns = New_Table_Columns_
                
                # For creating, create table if not default
                # Below line  is hide your warning 
                cursor.execute("SET sql_notes = 0; ")
                # create table here....
            
                exec_stmt = str("CREATE TABLE IF NOT EXISTS " + self.table_name + " (date DATETIME, username VARCHAR(20), tweet VARCHAR(280))")  
                
                cursor.execute(exec_stmt)
                db_connection.commit()

            def on_data(self, data):
                if self.num_tweets < self.max_tweets:
                    all_data = json.loads(data)
                    tweet = all_data["text"]
                    username = all_data["user"]["screen_name"]
                    
                    cur_time = datetime.datetime.strptime(all_data["created_at"], "%a %b %d %H:%M:%S %z %Y")
                    
                    exec_stmt = str("INSERT INTO " + self.table_name + " (date, username, tweet) VALUES (%s, %s, %s)")
                                
                    cursor.execute(exec_stmt, (cur_time, username, tweet))
                    
                    db_connection.commit()
                    
                    if self.num_tweets%tweet_add_milestone == 0:    
                        print("Successfully added tweet. Number:", self.num_tweets +1)
                    self.num_tweets +=1

                    return True
        
                else:
                    print("Finished writing to table:", self.table_name)
                    
                    return False 
                                          
            def on_error(self, status):
                print("Error Code:", status)
    
    
    #Initialise the stream listener
    listener = Stream_Listener(Max_Tweets_ = Max_Tweets, Table_Name_ = Table_Name, New_Table_Columns_ = New_Table_Columns, Tweet_Data_Parts_ = Tweet_Data_Parts)    
    #Authenticate the listener
    data_stream = tweepy.Stream(auth, listener)
    
    #Add filters
    data_stream.filter(track = Filters)
    
    
    # ## Read the tweets database  to Pandas
    # First create the engine to connect to the database
    engine = sqlalchemy.create_engine('mysql+mysqldb://root:pass@localhost/tweet_store')
    #Set up a metadata object to track table metadata
    meta_data = sqlalchemy.MetaData()
    tweet_table = sqlalchemy.Table(Table_Name, meta_data, autoload=True, autoload_with=engine)
    #Establish the database connection
    connection = engine.connect()
    #Create the query and execute it
    stmt = sqlalchemy.select([tweet_table])
    results = connection.execute(stmt).fetchall()
    
    df = pd.DataFrame(results)

    if Temporary:
            exec_stmt = "DELETE FROM " + Table_Name
            cursor.execute(exec_stmt)
            db_connection.commit()   

            connection.execute("FLUSH TABLES "+ Table_Name)
            
            stmt = "SELECT COUNT(*) FROM " + Table_Name
            num_rows = connection.execute(stmt).scalar()
           
            if num_rows ==0:
                print("Table Deleted!")
    else:
        pass

    return df


"""
============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

"""





def Stream_Twitter_Data_MYSQL(C_Key, C_Secret, A_Token, A_Token_Secret, Max_Tweets= 100, Filters = ["AAPL"], Table_Name = "tmp_tweets", New_Table_Columns = None, Tweet_Data_Parts = None, *args, **kwargs):

    """
   
    C_Key,
        String - Twitter Consumer Key
    C_Secret, 
        String - Twitter Consumer Secret
    A_Token,
        String - Twitter Access Token
    A_Token_Secret, 
        String - Twitter Access Token Secret
    Max_Tweets= 100,
        INT - Number of tweets to extract
    Filters = None, 
        List(String) - What words to filter on
        Default - ["Trump"]
    Table_Name = None,
        String - The name of your new table (default is tmp_tweets)
    New_Table_Columns = "(date DATETIME, username VARCHAR(20), tweet VARCHAR(280))",
        List(String) - SQL format tuple of string pairs for column name and type e.g. ['time DATETIME', 'age INT(2)']'
    Tweet_Data_Parts = None
        List(String/List(String)) - Parts of the tweet json (according to twitter) to extract e.g. [{"user":"screen_name"}, text'] is default
        Time is automatically added in to database
    Temporary = True,
        Bool - Store Tweets in the Database temporarily or permanently
        Default = True
    
    """
    exit_code = subprocess.check_call(["mysql.server", "start"])
    if exit_code ==0:
        pass

    else:
        raise Warning("Mysql server did not start, may want to start server manually")


    time.sleep(5)

    auth = tweepy.OAuthHandler(consumer_key=C_Key, consumer_secret=C_Secret)
    auth.set_access_token(A_Token, A_Token_Secret)

    db_connection = MySQLdb.connect("localhost","root", "pass", "tweet_store", charset = 'utf8mb4')
    cursor = db_connection.cursor()

    tweet_add_milestone = int(Max_Tweets/5)

     # ## Define a class to listen to the twitter API
    # If we want to use twitter data and/or a database other than the default then define this custom listener:
      
    class Stream_Listener(tweepy.StreamListener):
            def __init__(self, api=None, Max_Tweets_=None, Table_Name_=None, New_Table_Columns_=None, Tweet_Data_Parts_ = None):
                super().__init__()
                self.num_tweets = 0
                self.max_tweets = Max_Tweets_        
                self.table_name = Table_Name_
                self.tweet_data_parts = Tweet_Data_Parts_
                self.new_table_columns = New_Table_Columns_
                
                # For creating, create table if not default
                # Below line  is hide your warning 
                cursor.execute("SET sql_notes = 0; ")
                # create table here....
            
                exec_stmt = str("CREATE TABLE IF NOT EXISTS " + self.table_name + " (date DATETIME, username VARCHAR(20), tweet VARCHAR(280))")  
                
                cursor.execute(exec_stmt)
                db_connection.commit()

            def on_data(self, data):
                if self.num_tweets < self.max_tweets:
                    all_data = json.loads(data)
                    tweet = all_data["text"]
                    username = all_data["user"]["screen_name"]
                    
                    cur_time = datetime.datetime.strptime(all_data["created_at"], "%a %b %d %H:%M:%S %z %Y")
                    
                    exec_stmt = str("INSERT INTO " + self.table_name + " (date, username, tweet) VALUES (%s, %s, %s)")
                                
                    cursor.execute(exec_stmt, (cur_time, username, tweet))
                    
                    db_connection.commit()
                    
                    if self.num_tweets%tweet_add_milestone == 0 or self.num_tweets ==0:    
                        print("Successfully added tweet. Number:", self.num_tweets +1)
                    self.num_tweets +=1

                    return True
        
                else:
                    print("Finished writing to table:", self.table_name)
                    
                    return False 
                                          
            def on_error(self, status):
                print("Error Code:", status)


        #Initialise the stream listener
    listener = Stream_Listener(Max_Tweets_ = Max_Tweets, Table_Name_ = Table_Name, New_Table_Columns_ = New_Table_Columns, Tweet_Data_Parts_ = Tweet_Data_Parts)    
    #Authenticate the listener
    data_stream = tweepy.Stream(auth, listener)
    
    #Add filters
    data_stream.filter(track = Filters)


    subprocess.check_call(["mysql.server", "stop"])

    print("Database Server Successfully written to and MySQL server stopped")

    # First create the engine to connect to the database
    engine = sqlalchemy.create_engine('mysql+mysqldb://root:pass@australia90@localhost/tweet_store')
    #Set up a metadata object to track table metadata
    meta_data = sqlalchemy.MetaData()
    tweet_table = sqlalchemy.Table(Table_Name, meta_data, autoload=True, autoload_with=engine)
    #Establish the database connection
    connection = engine.connect()

    return db_connection
