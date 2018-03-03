import pandas as pd
import numpy as np
from math import log10, floor
import re


def Extract_Numeric_Column(dataframe_old , dataframe_new):
            """Take all float/integer columns and join to new dataframe"""
            dataframe_new = dataframe_new.reindex(dataframe_old.index)
            from math import log10, floor
            for series in dataframe_old:
                if (dataframe_old.dtypes[series].name == "int64") or(dataframe_old.dtypes[series].name == "float64"):
                        dataframe_new[series] = dataframe_old[series].copy()
                else:
                    pass
            return dataframe_new

def Extract_Categoricals(dataframe_old, dataframe_new):
            dataframe_new = dataframe_new.reindex(dataframe_old.index)
            for series in dataframe_old:
                if dataframe_old[series].dtypes.name == "category":
                    dataframe_new[series] = dataframe_old[series].copy()
                else:
                    pass
            return dataframe_new

def Datetime_to_Magnitude(dataframe_old , dataframe_new, Date_Series = None, Comparison_Date = "today"):
            """Takes all columns that are already datatimes or selcted and converts them into a magnitude in a new dataframe"""
            dataframe_new = dataframe_new.reindex(dataframe_old.index)
            if Date_Series != None:
                Comparison_Date = pd.to_datetime(Comparison_Date)
                import re
                pattern = re.compile("datetime([6432]{2})?(\[ns\])?|Timestamp")
                def subtract_dates(date_stamp, comparison_date = None):
                    delta_time = (comparison_date - date_stamp) /np.timedelta64(1, "D")
                    return delta_time
                if Date_Series == "All":
                    Date_Series = dataframe_old.columns.values
                else:
                    pass
                for series in Date_Series:
                    if pattern.match(dataframe_old.dtypes[series].name):
                         Delta_Time = dataframe_old[series].copy().apply(subtract_dates, comparison_date = Comparison_Date)
                         dataframe_new[series] = Delta_Time
                    else:
                        pass
            else:
                pass
            return dataframe_new



def Df_Series_to_Categorical(dataframe_old, dataframe_new, Cat_Series , max_categories):
            """Turns all reasonable series into dummies and appends to a a dataframe that's empty or same length as original data"""
            #Initialise new Dataframe to store result to
            #Make sure it has the correct index
            dataframe_new = dataframe_new.reindex(dataframe_old.index)
            if Cat_Series != None:
                for series in Cat_Series:
                    if dataframe_old[series].nunique() < max_categories:
                        #Preprocess the series
                        preprocessed_series = dataframe_old[series].str.lower()
                        preprocessed_series = preprocessed_series.str.split(pat = " ").str.join("_")
                        #Create Dummies
                        dummies = pd.get_dummies(preprocessed_series , drop_first = False, prefix = str(series + "_dummy"))
                        #Append dummies
                        dataframe_new = dataframe_new.join(dummies)
                    else:
                        #Don't want a sample that has too many categories such that in distance space points lie far away
                        print("Too many categories in:", series)
                        continue
            else:
                pass
            return dataframe_new


def extract_nums_in_series(element , number_pos = 0):
    """Extracts the i'th positive/negative integer/float from an element and returns it

    To extract from a pandas series then use the apply method (.apply(extract_nums_in_series)))

    """
    import re
    Extracted = re.findall(pattern = "[-\d.]+", string = str(element))
    return np.float(Extracted[number_pos])            


def Columns_to_Numeric(dataframe_old, dataframe_new , Num_Series):
            """Extracts all Targeted columns, turns to numeric and appends to dataframe """
            dataframe_new = dataframe_new.reindex(dataframe_old.index)
            if Num_Series != None:
                for series in Num_Series:
                    numeric_series = pd.to_numeric(dataframe_old[series], errors = "coerce")
                    dataframe_new = dataframe_new.join(numeric_series)
            else:
                pass
            return dataframe_new



def Extract_ith_Nums(dataframe_old, dataframe_new , Messy_Num_Series , Number_Pos = 0):
            """Takes a pandas.DataFrame.Series and returns the i'th number located within each element as a new column in a new DataFrame"""
            dataframe_new = dataframe_new.reindex(dataframe_old.index)
            if Messy_Num_Series != None:
                import re
                def extract_nums_in_series(element):
                    """Extracts the i'th positive/negative integer/float from an element and returns it"""
                    Extracted = re.findall(pattern = "[-\d.]+", string = str(element))
                    return Extracted[Number_Pos]
                for series in Messy_Num_Series:
                    Extracted = dataframe_old[series].apply(extract_nums_in_series).copy()
                    Extracted = Extracted.astype("float", errors = "ignore")
                    dataframe_new = dataframe_new.join(Extracted)
            else:
                pass
            return dataframe_new

       

            