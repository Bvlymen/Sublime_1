import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as TREE
import sys
from Random_Forest_Class import Random_Forest_Classifier
from sklearn.svm import SVC





def Plot_Decision_Boundaries_2D(X1, X2, y , Estimators, Test_Size = 0.3, Random_State = None, Scale = True , Colour_Map = plt.cm.coolwarm, Bright_Colour_Map = plt.cm.coolwarm, Alpha_Train = 1, Alpha_Test = 0.6, Certainty_Threshold = None, Variable_Names = ("Variable1","Variable2"), Delta = 0.02, Output = 1):
    
    def Return_Most_Certain_Classification_Data(X, y, Model, Certainty_Thresh = 0, Fit_First = False):
    
        if Fit_First:
            Model = Model.fit(X,y)
        if hasattr(Model, "predict_proba"):
            probabilities = Model.predict_proba(X)
        elif hasattr(Model, "decision_function"):
            probabilities = Model.decision_function(X)
        certainty_bool = np.amax(probabilities,axis = 1) > Certainty_Thresh
        
        certain_predictors, certain_response = X[certainty_bool], y[certainty_bool]
        print("Old number of samples:", len(y))
        print("New number of samples:", len(certain_response))
      
        return certain_predictors, certain_response

    if (Certainty_Threshold != None) & (isinstance(Estimators, list)):
            X_Combined = np.hstack((X1.reshape(-1,1),X2.reshape(-1,1)))
            X, y = Return_Most_Certain_Classification_Data(X_Combined, y, Model = Estimators[0], Certainty_Thresh = Certainty_Threshold, Fit_First= True)
            X1, X2 = X[:,0], X[:,1]

    if Certainty_Threshold != None:
            X_Combined = np.hstack((X1.reshape(-1,1),X2.reshape(-1,1)))
            X, y = Return_Most_Certain_Classification_Data(X_Combined, y, Model = Estimators, Certainty_Thresh = Certainty_Threshold, Fit_First= True)
            X1, X2 = X[:,0], X[:,1]
    
    #Define a class bijection for class colour mapping
    unique_classes, y_bijection = np.unique(y, return_inverse = True)
   
    #Sort the data so colour labels match up with actual labels
    X1 , X2 = X1.reshape((-1,1)), X2.reshape((-1,1))
    y_bijection = y_bijection.reshape((-1,1))
    

    Full_combined = np.hstack((X1,X2, y_bijection))
    Full_combined = Full_combined[Full_combined[:,2].argsort()]

    X1 , X2 = Full_combined[:,0].reshape((-1,1)), Full_combined[:,1].reshape((-1,1))
    y_bijection = Full_combined[:,2].reshape((-1,1))
    
    #Preprocess the data if needed:
    X1, X2 = StandardScaler().fit_transform(X1), StandardScaler().fit_transform(X2)

    delta = Delta #Step size in the mesh

    figure = plt.figure(figsize = (12,8))

    x1_min, x1_max = X1.min() -0.5, X1.max() +0.5
    x2_min, x2_max = X2.min() -0.5, X2.max() +0.5

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, delta), np.arange(x2_min, x2_max, delta))

    
    #Plot the given data (colourmap)

    col_map = Colour_Map
    col_map_bright = Bright_Colour_Map


    #Ready a train test split
    Full_combined = np.hstack((X1, X2, y_bijection))
    

    X_train, X_test, y_train, y_test = train_test_split(Full_combined[:,[0,1]], Full_combined[:,2], test_size = Test_Size, random_state = Random_State)

    #Get a figure and axes based on how many estimators (1 or multiple there are)
    #Multiple estimators
    if isinstance(Estimators, (list, np.ndarray)):
        n_rows = len(Estimators)
    

        fig, axes = plt.subplots(nrows = n_rows, ncols= 2, sharex= True, sharey= True, figsize = (12,n_rows*4)) 
    #One estimator
    else:
        Estimators = np.array([Estimators])
        fig, axes = plt.subplots(1,2, figsize = (12,8))
        axes = np.array([axes])
    
    for axs, Estimator in zip(axes[:], Estimators):
        
        ax1, ax2 = axs[0], axs[1]
        
        ax1.set_title("Input Data")
        #Plot Training data
        scat = ax1.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = col_map_bright, edgecolors = 'k', alpha= Alpha_Train)
        #And testing data
        ax1.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = col_map_bright, edgecolors = 'k', alpha =Alpha_Test)

        ax1.set_xlim(xx.min(), xx.max())
        ax1.set_ylim(yy.min(), yy.max())

    
        
        ax1.set_xlabel(Variable_Names[0])
        ax1.set_ylabel(Variable_Names[1])


        #Now for the classifier

        model = Estimator.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        #Plot the decision boundary. For that, we will assign a colour to each point 
        # in the mesh [x1_min, x1_max]*[x2_min, x2_max]
        
        if hasattr(model, "decision_function"):
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            if len(Z.shape) >1:
                Z = np.argmax(Z, axis = 1)
            else:
                mapper = lambda x: 0 if x < 0 else 1
                mapper_func = np.vectorize(mapper)
                Z = np.sign(Z)
                Z = mapper_func(Z)
                

        elif hasattr(model, "predict_proba"):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z = np.argmax(Z, axis = 1)
        
        else:
            print("This Estimator doesn't have a decision_function attribute and can't predict probabilities")

     
        Z_uniques = np.unique(Z.astype(int))

        unique_predictions = unique_classes[Z_uniques]
        
        #Put the result in a colourplot

        Z = Z.reshape(xx.shape)
        
        contour = ax2.pcolormesh(xx, yy, Z, vmin = Z.min(), vmax= Z.max(), cmap = col_map, alpha=0.7)

        #Plot also the training data
        ax2.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = col_map_bright, edgecolors = 'k', alpha= Alpha_Train)
        #And testing data
        ax2.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = col_map_bright, edgecolors = 'k', alpha = Alpha_Test)

        ax2.set_xlim(xx.min(), xx.max())
        ax2.set_ylim(yy.min(), yy.max())

        
        ax2.set_xlabel(Variable_Names[0])
        ax2.set_ylabel(Variable_Names[1])
        # ax2.set_title(str(Estimator))

        ax2.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

        cb1 = plt.colorbar(scat, spacing = "proportional", ax = ax1, ticks = np.arange(len(unique_classes)))
        cb1.ax.set_yticklabels(unique_classes)

        print("Unique Predictions: {}".format(unique_classes[Z_uniques]), "for: {}".format(Estimator))
        
        ticks = np.linspace(Z.min(), Z.max(), len(unique_predictions))

        cb2 = plt.colorbar(contour, spacing = "proportional", ax=ax2, ticks = ticks)
        cb2.ax.set_yticklabels(unique_predictions)
        
        #Also print the score of the model
        print("Model Score:",score, "\n")

    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    fig.suptitle("Data and Classification Boundaries", fontsize =20)

    if Output == 1:
        return fig
    elif Output == 2:
        return model, fig
    else:
        return fig