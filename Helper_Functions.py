import pprint
import sys
import argparse
import re
import stat
import time
import datetime
import platform
import inspect
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error
import operator
import statsmodels.api as sm

#ECDFs function:
def plot_ecdf(x):
    """Create and add Plot of an Exploratory CDF for a given np.array of data"""
    x= np.sort(x)
    n=len(x)
    y = np.arange(1,n+1)/n
    _ = plt.plot(x,y, linestyle="none", marker=".")

#Debugging functions:
#Insert on line of code and enter objects within code as arguments to check what is assigned to them/if code is being reached
def warning(*objs):
    """A print statement for any objects existing at a certain point in a function used for debugging"""
   
    print("WARNING: ", os.path.basename(inspect.currentframe().f_back.f_globals['__file__']), inspect.currentframe().f_back.f_lineno, *objs, file=sys.stderr,  sep=' ') # note use of line number information from the previous stackframe

#Same as warning but can turn on and off with verbosity
def information(verbosity, *objs):

    if verbosity:
        print("Information: ", os.path.basename(inspect.currentframe().f_back.f_globals['__file__']), inspect.currentframe().f_back.f_lineno, *objs, file=sys.stderr,  sep=' ') # note use of line number information from the previous stackframe



#Plot Residuals vs. OLS Variable or alternative variable
def Plot_Residuals_Against_Variable(Predictor_train, Response_train, Alt_Variable_Name = "predictors", Estimator = LinearRegression, Predictor_test = "train", Response_test = "train" , Alt_Variable = "predictors"):
    """Forms residuals of the predictions from a model and plots them vs. an alternative variable"""
    

    if isinstance(Predictor_train, type(pd.Series())):
        Predictor_train = Predictor_train.to_frame()

    Model = Estimator()
    Model.fit(Predictor_train, Response_train)

    if Response_test == "train":
        Response_test = Response_train

    if Predictor_test == "train":
        Predictor_test = Predictor_train

    Predictions = Model.predict(Predictor_test)

    residuals = Response_test - Predictions

    if Alt_Variable == "predictors":
        if len(Predictor_train.columns) > 1:
            Alt_Variable = Predictor_train.loc[:,Alt_Variable_Name]
        else:
            Alt_Variable = Predictor_train

    ax0 = plt.figure(1)

    ax0 = plt.plot(Alt_Variable, residuals, "r.")
    ax0 = plt.title("Estimator Residuals vs. Alt_Variable")
    ax0 = plt.xlabel("Alternative Variable")
    ax0 = plt.ylabel("Residuals")

    return  ax0


def Plot_Variables_In_3D(X, Y, Z=0, **kwargs):
    """Plots 3 Series of variables in 3D"""

    fig = plt.figure()
    ax0 = fig.add_subplot(111, projection = "3d")

    ax0.scatter(xs = X, ys = Y, zs = Z, **kwargs)

    ax0.set_xlabel("X_Variable")
    ax0.set_ylabel("Y_Variable")
    ax0.set_zlabel("Z_Variable")

    return ax0



#Adjusted R_Squared:
# ver.0
def adj_r2_score0(Model, y, y_predictions):
    """Return adjusted R-Squared for model, response and predictions"""
 
    adj = 1 - float(len(y)-1)/(len(y)-len(Model.coef_)-1)*(1 -
    r2_score(y,y_predictions))
    return adj

#ver.1
def adj_r2_score1(Model, X,y):
    """Return adjusted R-Squared for model, response and predictions"""
    Model.fit(X,y)
    adj = 1 - float(len(y)-1)/(len(y)-len(Model.coef_)-1)*(1 -
    Model.score(X,y))
    return adj



def Compare_Interaction(Series1, Series2, Response, Estimator = LinearRegression, Interaction_Type = "*", Score = "r2_score"):
    """Compare the perfomance of an interaction term in a model to without it"""
    def adj_r2_score(model, y, yhat):
        adj = 1 - float(len(y)-1)/(len(y)-len(model.coef_)-1)*(1 -r2_score(y,yhat))
        return adj

    interaction_dict = {"*": operator.mul, "+": operator.add, "/": operator.truediv , "reciprocal": operator.inv, "**": operator.pow}

    Model_NI = Estimator()
    Model_I = Estimator()

    interaction_series = interaction_dict[Interaction_Type](Series1, Series2)

    no_interaction_predictors = pd.concat([Series1, Series2], axis = "columns")
    interaction_predictors = pd.concat([no_interaction_predictors, interaction_series], axis = "columns")

    Model_NI.fit(no_interaction_predictors, Response)
    Model_I.fit(interaction_predictors, Response)


    NI_predictions = Model_NI.predict(no_interaction_predictors)
    I_predictions = Model_I.predict(interaction_predictors)

    score_dict = {"r2_score":r2_score, "mean_squared_error":mean_squared_error}

    if Score == "adj_r2_score":
        I_score = adj_r2_score(Model_I, Response, I_predictions)
        NI_score = adj_r2_score(Model_NI, Response, NI_predictions)
    else:
        I_score = score_dict[Score](Response, I_predictions)
        NI_score = score_dict[Score](Response, NI_predictions)

    print("Non-Interaction Score: {}".format(NI_score))
    print("Interaction Score: {}".format(I_score))




def Fit_ith_Order_Polynomial(Dataframe, Response,  Estimator = LinearRegression, Polynomial_Order = 2):
    """Fit an i'th order Polynomial and print the score of the model and return the model"""
    variables = pd.DataFrame(index = Dataframe.index)

    Model = Estimator()

    if type(Polynomial_Order) is int:
        for i in range(Polynomial_Order):
            variables = variables.join(Dataframe ** (Polynomial_Order + 1), rsuffix = str(i+1))

    elif type(Polynomial_Order) is dict:
        for k, v in Polynomial_Order.items():
            for i in range(v):
                variables = variables.join(Dataframe[k]** (v+1), rsuffix = str(i+1))

    else:
        raise ValueError("Polynomial_Order must be of type int64 or dict")

    Model.fit(variables, Response)

    print(Model.score(variables, Response))

    return Model



def Plot_Polynomial(Predictor, Response, Estimator = LinearRegression, Order = 1):
    """Plot an i'th order polynomial for a variable against a response and show the Estimator line"""

    if not isinstance(Predictor, type(pd.DataFrame())):
        Predictor = Predictor.to_frame()
    if not isinstance(Response, type(pd.DataFrame())):
        Response = Response.to_frame()

    poly_predictors = pd.DataFrame(index = Predictor.index)

    Model = Estimator()

    for i in range(Order):
        poly_predictors = poly_predictors.join(Predictor ** (i+1), rsuffix = str(i+1))

    Model.fit(poly_predictors , Response)

    X_line = np.linspace(min(Predictor.iloc[:,0]), max(Predictor.iloc[:,0]), 500)
    X_line_predictors = np.array([X_line **(i+1) for i in range(Order)]).transpose()


    Y_line = Model.predict(X_line_predictors)

    ax0 = plt.plot(X_line, Y_line, "r-")
    ax0 = plt.plot(Predictor, Response, color = "navy", linestyle = "None", marker = ".")
    ax0 = plt.xlabel("Predictor")
    ax0 = plt.ylabel("Response")

    print(Model.score(poly_predictors, Response))

    return ax0


def Scale_Variables(DataFrame, Columns = "All"):
    """Scale selected Columns by Mean and Standard Deviation"""
    if Columns == "All":
        Columns = DataFrame.columns.values
    else:
        pass
    dataframe = DataFrame.copy()
    dataframe[Columns] =  dataframe[Columns].transform(lambda x: (x - x.mean())/x.std())
    return dataframe


def Transform_Variables(DataFrame, Columns = "All", Transformation = np.log):
    if Columns == "All":
        Columns = DataFrame.columns.values
    dataframe = DataFrame.copy()
    if Transformation == np.log or np.log1p:
        negative_mask = dataframe[Columns]<0

        df = dataframe[Columns].copy()
        df = abs(df)

        df = df.transform(Transformation)

        df[negative_mask] = df[negative_mask] * -1
        dataframe[str(Columns + "_transformation")] = df

    else:
        dataframe[str(Columns + "_transformation")] = dataframe[Columns].transform(Transformation)
    return dataframe


def Draw_Bs_Pairs_Replicates(x, y, func, size=1, **kwargs):
    """Create an array of Bootstrap Replicates for a given function"""

    shape_of_statistics = func(x, y, **kwargs).shape[0]

    bs_replicates = np.ndarray(shape = (shape_of_statistics,0))
    indicies = np.arange(len(x))
    for i in range(size):
        
        bs_indicies = np.random.choice(indicies, size = len(indicies))
        bs_x, bs_y = x[bs_indicies], y[bs_indicies]
        
        statistics = func(bs_x, bs_y, **kwargs).reshape((-1,1))
        
        bs_replicates = np.hstack((bs_replicates, statistics))
        
    return bs_replicates

def Merge_2_Variables_By_Index(Series1, Series2, index1= None, index2=None, How_Join = "inner"):
    if type(Series1) == type(pd.Series()):
        Series1 = Series1.to_frame()
    if type(Series2) == type(pd.Series()):
        Series2 = Series2.to_frame()

    if (index1 ==None) and (index2 == None):
        frame = pd.merge(Series1,Series2, left_index = True, right_index =True, how = How_Join )

    elif (index1 != None) and (index2 != None):
        frame = pd.merge(Series1,Series2, left_on = index1, right_on = index2, how = How_Join )
    
    elif (index1 == None) and (index2 != None):
        frame = pd.merge(Series1,Series2, left_index = True, right_on = index2, how = How_Join )
    
    elif (index1 != None) and (index2 == None):
        frame = pd.merge(Series1,Series2, left_on = index1, right_index = True, how = How_Join )

    else:
        pass    

    return frame




# def Plot_3D_Regression(X, y):

#     def Plot_Variables_In_3D(X, Y, Z=0, **kwargs):
#     """Plots 3 Series of variables in 3D"""

#     fig = plt.figure()
#     ax0 = fig.add_subplot(111, projection = "3d")

#     ax0.scatter(xs = X, ys = Y, zs = Z, **kwargs)

#     ax0.set_xlabel("X1: predictor")
#     ax0.set_ylabel("X2: predictor")
#     ax0.set_zlabel("y: response")

#     return ax0

#     reg = sm.OLS(endog = y, exog = sm.add_constant(X)).fit()

#     x_space = np.linspace(X)
#     y_line = 



def Encode_Categories_To_Numbers(DataFrame, Categorical_Features= None, Max_Categories = 20):
    """

    Dataframe = pandas.DataFrame: df  to encode categories to numbers for
    Categorical_Features = List: list of features to convert
    Max_Categories = Float/int: Number of categories above which to temove features for 

    """

    if Max_Categories == None:
        Max_Categories =np.inf
   
    #Return a list of feature names over the max number of categories
    def check_num_categories(dataframe_, max_categories_):
        #Return a series with counts of unique categories for each feature
        num_of_cats = dataframe_.nunique()
        
        #Find the labels of features with too many categories
        labels_over_max = num_of_cats.index[num_of_cats.values>max_categories_]

        return labels_over_max

    #Modify the dataframe we are working with to only contain categorical features
    if Categorical_Features:
        dataframe = DataFrame[Categorical_Features]

    else:
        dataframe = DataFrame

    features_to_drop = check_num_categories(dataframe_ =dataframe, max_categories_ = Max_Categories)

    #Drop the features with too many categories
    dataframe = dataframe.drop(features_to_drop, axis =1)

    #Map the remaining feature categories to integers representing that category
    def map_cats_to_nums(series_):
        
        #Convert Series to right format
        series_  = series_.astype(str)
       
        #Find how many unique categories there are
        nunique = series_.nunique()

        #Find what those unique categories are and make sure they are of the right type for mapping
        uniques = list(series_.unique())
        #Integers to map categories to for each series
        mapped = np.arange(nunique).astype(int)

        #Map each unique category to a unique number
        mapping_dict = dict(zip(uniques, mapped))

        return series_.map(mapping_dict)

    #Do this for the whole dataframe
    dataframe = dataframe.apply(map_cats_to_nums)

    return dataframe







    




