#To run the app, on the commad line run:
# $ bokeh serve --allow-websocket-origin=localhost:5006 --port=5006 ~/Downloads/Decision_Boundaries_App.py --show
# This file must be in your downloads folder along with the dataset: "wage2.dta"
# If the browser doesn't open to display the app then type into your browser: "Localhost:5006"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import train_test_split

from bokeh.plotting import gridplot, figure, show
from bokeh.models.tools import HoverTool, LassoSelectTool, WheelZoomTool
from bokeh.models.widgets.buttons import Button
from bokeh.models.widgets import Select, MultiSelect
from bokeh.models import ColumnDataSource, ColorBar, BasicTicker
from bokeh.models.glyphs import Text
from bokeh.palettes import RdBu, inferno
from bokeh.models.mappers import LinearColorMapper

from bokeh.layouts import gridplot, widgetbox, column, row, layout
from bokeh.io import curdoc, output_file

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application

import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as TREE

import sys
sys.path.append("/Users/ben/Desktop/Python/Sublime/")
sys.path.append("/Users/ben/Desktop/Python/Jupyter/Econometrics_IIA/")
sys.path.append("~/Downloads/")

from sklearn.svm import SVC



# Steps:
# # 1. Bijection for y
# 2. Fit models 
# 3. Create ranges for figure and the initial figure
# 4. For each model predict the class and create the colour plot
# 5. Organise the plots
# 5. add the tooltips
# 6. add the dropdowns and button to update
# 7. enable functionality for changing variables involving defining function for 2-6
#     involves defining the plotted images (pi, pj,...), the ranges, the tootltips, the new fitted models, the predictions, the colourmesh



X=pd.read_stata("wage2.dta")
X["bin_lwage"] = pd.qcut(X.lwage,q=10, labels = False)
Variable1 = "IQ"
Variable2 = "KWW"
y="bin_lwage"
Estimators = [LDA(),QDA(),KNN(),GNB(),TREE(),SVC(probability=True)]

Test_Size =0.3
Random_State = 42
Scale = True
Palette = "RdBu"
Delta = 0.02
Output = 1
Estimator_Names = [str(estimator).split("(")[0] for estimator in Estimators]




    
def create_plots(Estimators):

    print("Starting update")

    if not isinstance(Estimators, (type(np.array), list)):
        Estimators = np.array(Estimators)

    estimator_names = np.array(list(estimator_select.value))
    ix = np.isin(Estimator_Names, estimator_names)
    estimator_indices =  [int(i) for i in np.where(ix)[0].flatten()]

    estimators = np.array(Estimators)[estimator_indices]
    
    variable1 = drop1.value
    variable2 = drop2.value
    y = drop3.value
    
    #Things to update:
    # image background i.e. image source √
    # observation source √
    #Color mapper values√
    #hover tool values √
    #Figure ranges √
    #Model score text things √

    #Lets calculate all the image and observation data first 

   
    plots = [None for i in range(len(estimators))]
    image_sources = [None for i in range(len(estimators))]
    observation_sources = [None for i in range(len(estimators))]
    hover_tools = [None for i in range(len(estimators))]
    model_score_sources= [None for i in range(len(estimators))]
    glyphs0= [None for i in range(len(estimators))]
    color_bars= [None for i in range(len(estimators))]
    p_circles = [None for i in range(len(estimators))]
    p_images = [None for i in range(len(estimators))]
      

    #Iterate over the estimators
    for idx, estimator in enumerate(estimators):
        #Find the title for each plot
        estimator_name = str(estimator).split('(')[0]
        
        #Extract the needed data
        full_mat = X[[variable1, variable2, y]].dropna(how = "any", axis = 0)

        #Define a class bijection for class colour mapping
        unique_classes, y_bijection = np.unique(full_mat[y], return_inverse = True)
        full_mat['y_bijection'] = y_bijection
        
        #Rescale the X Data so that the data fits nicely on the axis/predictions are reliable
        full_mat[variable1 + "_s"] = StandardScaler().fit_transform(full_mat[variable1].values.reshape((-1,1)))
        full_mat[variable2 + "_s"] = StandardScaler().fit_transform(full_mat[variable2].values.reshape((-1,1)))

        #Define the Step size in the mesh
        delta = Delta 

        #Separate the data into arrays so it is easy to work with
        X1 = full_mat[variable1 + "_s"].values
        X2 = full_mat[variable2 + "_s"].values
        Y = full_mat["y_bijection"].values 

        #Define the mesh-grid co-ordiantes over which to colour in
        x1_min, x1_max = X1.min() -0.5, X1.max() +0.5
        x2_min, x2_max = X2.min() -0.5, X2.max() +0.5

        #Create the meshgrid itself
        x1, x2 = np.arange(x1_min, x1_max, delta), np.arange(x2_min, x2_max, delta)
        x1x1, x2x2 = np.meshgrid(x1, x2)

        #Create the train test split
        X_train, X_test, y_train, y_test = train_test_split(full_mat[[variable1+"_s",variable2+"_s"]], Y, test_size = Test_Size, random_state = Random_State)
        #Fit and predict/score the model
        model = estimator.fit(X= X_train, y= y_train)
        # train_preds = model.predict(X_train)
        # test_preds = model.predict(X_test)
        model_score = model.score(X_test, y_test)
        model_score_text = "Model score: %.2f" % model_score

        if hasattr(model, "decision_function"):
            Z = model.decision_function(np.c_[x1x1.ravel(), x2x2.ravel()])

        elif hasattr(model, "predict_proba"):
            Z = model.predict_proba(np.c_[x1x1.ravel(), x2x2.ravel()])
        
        else:
            print("This Estimator doesn't have a decision_function attribute and can't predict probabilities")

     
        Z = np.argmax(Z, axis = 1)  
        Z_uniques = np.unique(Z)

        unique_predictions = unique_classes[Z_uniques]

        Z = Z.reshape(x1x1.shape)

        #Add in the probabilities and predicitions for the tooltips
        full_mat["probability"] = np.amax(model.predict_proba(full_mat[[variable1 + "_s", variable2 + "_s"]]), axis = 1)
        
        bijected_predictions= model.predict(full_mat[[variable1 + "_s", variable2 + "_s"]])
        full_mat["prediction"] = unique_classes[bijected_predictions]

        #Add an associated color to the predictions   
        number_of_colors= len(np.unique(y_bijection))
        
        #Create the hover tool to be updated
        hover = HoverTool(tooltips = [
            (variable1,"@"+variable1),
             (variable2, "@"+variable2),
              ("Probability", "@probability"),
               ("Prediction", "@prediction"),
               ("Actual", "@"+y)])
        
        #Create the axes for all the plots
        plots[idx] = figure(x_axis_label = variable1, y_axis_label = variable2, title = estimator_name, x_range = (x1x1.min(),x1x1.max()),y_range =  (x2x2.min(),x2x2.max()), plot_height = 600, plot_width = 600, toolbar_sticky = False, toolbar_location = "above")

        #Create all the image sources
        image_data = dict()
        image_data['x'] = np.array([x1x1.min()])
        image_data["y"] = np.array([x2x2.min()])
        image_data['dw'] = np.array([x1x1.max()-x1x1.min()])
        image_data['dh'] = np.array([x2x2.max() - x2x2.min()])
        image_data['boundaries'] = [Z]
        

        image_sources[idx] = ColumnDataSource(image_data)

        if number_of_colors > 11:
            col_palette = inferno(number_of_colors)
        elif number_of_colors >2:
            col_palette = RdBu[number_of_colors]
        else:
            col_palette = ["#1D61F2 ","#F73B28"]

        low = full_mat["y_bijection"].min()
        high = full_mat["y_bijection"].max()
        cbar_mapper = LinearColorMapper(palette = col_palette, high = high, low = low)

        p_images[idx] = plots[idx].image(image = 'boundaries', x= 'x', y = 'y', dw = 'dw', dh= 'dh', color_mapper = cbar_mapper, source = image_sources[idx])
        
        #Create the sources to update the observation points 
        observation_sources[idx] = ColumnDataSource(data = full_mat)  
        
        p_circles[idx] = plots[idx].circle(x =variable1 +"_s", y= variable2 + "_s", color = dict(field = 'y_bijection', transform = cbar_mapper),  source = observation_sources[idx], line_color = "black")
       

        #Create the hovertool for each plot
        hover_tools[idx] = hover

        #Add the hover tools to each plot
        plots[idx].add_tools(hover_tools[idx])

        #Create all the text sources (model scores) for the plots
        model_score_sources[idx] = ColumnDataSource(data = dict(x=[x1x1.min()+0.3], y=[x2x2.min()+0.3], text=[model_score_text]))

        #Add the model scores to all the plots
        score_as_text = Text(x = "x", y = "y", text = "text")
        glyphs0[idx] = plots[idx].add_glyph(model_score_sources[idx], score_as_text)

        #Add a colorbar
        color_bars[idx] = ColorBar(color_mapper= cbar_mapper , ticker=BasicTicker(desired_num_ticks = number_of_colors), label_standoff=12, location=(0,0), bar_line_color = "black")

        plots[idx].add_layout(color_bars[idx],"right")
        plots[idx].add_tools(LassoSelectTool(), WheelZoomTool())
        
        # configure so that no drag tools are active
        plots[idx].toolbar.tools = plots[idx].toolbar.tools[1:]       
        plots[idx].toolbar.tools[0], plots[idx].toolbar.tools[-2] = plots[idx].toolbar.tools[-2], plots[idx].toolbar.tools[0]
    
    layout_col = [row(plot) for plot in plots]
    print("Ending Update")
    return layout_col


def update():
    
    plot.children[1] = column(create_plots(Estimators))

    
 #lowecase innerscope variables
variable1, variable2 = Variable1, Variable2

#Create the widgets
drop1 = Select(title = "Variable 1", options = list(X.columns.values), value = variable1)
    
drop2 = Select(title = "Variable 2", options = list(X.columns.values), value = variable2)
    
drop3 = Select(title = "variable 3", options = list(X.columns.values), value = y)

estimator_names = [str(estimator).split("(")[0] for estimator in Estimators]
estimator_indices = [str(i) for i , name in enumerate(estimator_names)]
estimator_select = MultiSelect(title= "Estimators", options = estimator_names, value = estimator_names[0:5])
    
button = Button(label = "Update", button_type = "success")
button.on_click(update)

plot =row(widgetbox(drop1, drop2, drop3, estimator_select, button), column(create_plots(Estimators)))

curdoc().add_root(plot)
curdoc().title = "Wage Data Visualisation"

    
        

















