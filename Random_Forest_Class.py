from __future__ import division
import numpy as np
import pandas as pd

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as TREE
from sklearn.svm import SVC

import warnings

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import assert_all_finite
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.metrics import accuracy_score, precision_score, log_loss


class Random_Forest_Classifier(LDA):
	
	def __init__(self, Classifiers = [LDA, QDA, KNN, GNB, TREE], cv = 3, Bootstrap_Samples = 5, Rand_Param_Search = True, Rand_Search_Iterations = 5, Param_Dicts = "default", Random_State = None):
		
		self.Classifiers = Classifiers
		self.cv = cv
		self.Bootstrap_Samples = Bootstrap_Samples
		self.Rand_Param_Search = Rand_Param_Search
		self.Rand_Search_Iterations = Rand_Search_Iterations
		self.Param_Dicts = Param_Dicts
		self.Random_State = Random_State


	def fit(self, X, y):
		"""	A Function which Performs a random forest classification on a training set of data and returns the model + score for a test set of data

		X:
			pandas dataframe, series or numpy array shape (n x p)
		y:
			pandas dataframe, series or numpy array length n x 1 
		cv:
			nuber of cross validation splits for the data
		Bootstrap_Samples:
			Number of Bootstrap samples
		Rand_Param_Search:
			Boolean: True then use sklearn RandomizedSearchCV, False use GridSearchCV
		Rand_Search_Iterations:
			The numer of iterations in Randomised Search CV
		Param_Dicts:
			str. or list
			str = ["guessed", "default"]
			List of Dictionaries for each estimator in the same order as the classifiers.
			The Parameters to check in a Cross val hyper parameter search
			

		Steps:
		1. For each Bootstrap sample: tick
			2. For each model: tick
				3. Random/Grid Search CV tick
				4. Get Best Scorestick
		5. For each model take the median scores and modal paramters
		6. Fit the models
		7. Return a class that given an X and y of the same shape as previous input X,y will return their class predictions weighted by the median of the, best-model scores in fitting

		#Idea to include different scoring functions/one that weights each prediction equally or by some predefined amount
		"""	
		if isinstance(X, (pd.DataFrame,pd.Series)):
			X= X.values
		
		if isinstance(y, pd.Series):
			y= y.values

		#Initialise the bootstraps to be used for model testing
		bootstrap_samples = []
		bootstrap_indices = np.arange(len(X))
		
		#Create the list of bootstraps
		for i in range(self.Bootstrap_Samples):
			
			resample = np.random.choice(a= bootstrap_indices, replace = True, size = len(X))
			bootstrap_X = X[resample]
			bootstrap_y = y[resample]
			bootstrap_samples.append((bootstrap_X, bootstrap_y))

		if self.Param_Dicts == "guessed":
			
			GNB_dict = {"priors":[None]}
			LDA_dict = {"n_components":[None, 3, 5, 10,20,30,50]}
			QDA_dict = {"reg_param":np.linspace(0,1,8)}
			TREE_dict = {"criterion":["entropy", "gini"], "max_depth":[3,6,9,10,12], "min_samples_split":[10,20,30]}
			KNN_dict = {"n_neighbors":[5,10,20,35], "weights":["distance", "uniform"]}
			SVC_dict = "This feature needs implemented whereby there is a search and insert"

			param_dicts = [LDA_dict, QDA_dict, KNN_dict, GNB_dict, TREE_dict]

		elif self.Param_Dicts == "default":
			param_dicts = np.full((len(self.Classifiers),), {})
		
		else:
			param_dicts = self.Param_Dicts



		#initialise list of fully fit and tuned models
		fully_fit_models = []
		model_median_scores = []
		#For loop - for each classifier do things
		for estimator in self.Classifiers:
			#Find estimator name
			estimator_name = str(estimator()).split("(")[0]
			
			#Initialise the scores and parameters list
			exec(estimator_name + "_scores = []")
			exec(estimator_name + "_params = []" )

			#For loop - for each bootstrap calculate hyperparameters and scores
			for idx, (bootstrap, param_dict) in enumerate(zip(bootstrap_samples, param_dicts)):
				#Check whether we Randomse or grid search the hyperparameters
				if (self.Rand_Param_Search == True) & (self.Rand_Search_Iterations <= len(np.array(list(param_dict.values())).ravel())):
					try:
						#Fit the model to a Randomised search CV
						hyper_search_cv = RandomizedSearchCV(estimator = estimator(), param_distributions = param_dict, cv = self.cv, n_iter = self.Rand_Search_Iterations, iid = False, random_state = self.Random_State)
					
					except ValueError:
						try:
							assert continue_fit
							if (continue_fit == "y") | (continue_fit == "yes"):
								hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = cv, iid = False)
						
						except (NameError, UnboundLocalError):
							print("Most likely the Paramter Grid is too small for 'n_iter' in RandomizedSearchCV.")
							continue_fit = str(input("Can try to continue with GridSearchCV? [y/n]:"))
					
							if (continue_fit == "y") | (continue_fit == "yes"):
								try:
									hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = self.cv, iid = False)
								except:
									break
							
							else:
								print("Stopping")
								break
					except:
						break

				elif (self.Rand_Param_Search == True) & (self.Rand_Search_Iterations >= len(np.array(list(param_dict.values())).ravel())):
					
					try:
						assert continue_fit
						if (continue_fit == "y") | (continue_fit == "yes"):					
							hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = self.cv, iid = False)
						
					except (NameError, UnboundLocalError):

						print("Most likely Paramter Grid too small for n_iter in RandomizedSearchCV.")
						continue_fit = str(input("Can try to continue with GridSearchCV? [y/n]:"))
				
						if (continue_fit == "y") | (continue_fit == "yes"):
							hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = self.cv, iid = False)
						else:
							print("Stopping")
							break
				

				elif self.Rand_Param_Search == False:
					#Fit the model to a Grid search CV
					hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = self.cv, iid = False)
						
					
				#For invalid argument values
				else:

					raise ValueError("Invalid argument value. 'Rand_Param_Search' Should be of type bool.")

				#Fit the model and search for hyperparams
				
				hyper_search_cv.fit(X, y.ravel())
				

				best_model_params = hyper_search_cv.best_params_
				model_score = hyper_search_cv.best_score_

				exec(estimator_name + "_scores.append(model_score)")
				# exec(estimator_name + "_params.append(tuple(best_model_params.items()))")

				print("Completed Fitting of", estimator_name, "for bootstrap:", idx)
				
				if idx ==self.Bootstrap_Samples-1:
					print("Mean Score for", estimator_name+":",eval("np.mean("+estimator_name+"_scores)"))

			#Create a counter to find the most common hyperparameters (((EXTRA THINGS TO ADD LATER)))
		
			# exec("params_counter = Counter("+estimator_name+"_params)")
			# exec(estimator_name+ "_mode_hyperparams =  params_counter.most_common(1)")
			
			#Compute the median score for each model
			exec(estimator_name + "_median_score = np.median(" + estimator_name + "_scores)")
			#Fit each model with the found modal hyperparameters
			exec(estimator_name + "_full_fit = hyper_search_cv.best_estimator_.fit(X,y)")

			#Add the fitted models and median scores to a list of full models and median scores for each model
			exec("model_median_scores.append(" + estimator_name + "_median_score)")
			exec("fully_fit_models.append(" + estimator_name + "_full_fit)")

			#Just some verbosity
			print("Added a classifier:", estimator_name,"to the ensemble!")

		class Random_Forest_Model(object):
			"""
			The class for an ensemble classifier
			"""

			#Update class with the given fit models and scores (weights)
			def __init__(self, Models, Scores,X, y):
				self.models = Models 
				self.model_weights = Scores/sum(Scores)
				self.training_X = X
				self.training_y = y

			#Define the random forest prediction function
			def predict(self, X):
				
				if isinstance(X, (pd.DataFrame,pd.Series)):
					X= X.values

				list_of_pred_matrices=  []
				unique_classes = np.unique(self.training_y)

				for idx, (model, weight) in enumerate(zip(self.models, self.model_weights)):
					
					if hasattr(model, "predict"):
						
						model_pred = model.predict(X)
						model_pred = np.append(model_pred, np.unique(unique_classes))

						unique_preds, model_pred_nums = np.unique(model_pred, return_inverse = True)
						
						pred_len = len(model_pred_nums)

						binary_pred_matrix = np.zeros((len(model_pred_nums), len(unique_classes)))

						binary_pred_matrix[np.arange(pred_len), model_pred_nums] = 1

						binary_pred_matrix = binary_pred_matrix*weight
						
						list_of_pred_matrices.append(binary_pred_matrix)

					else:
						raise TypeError("Model doesn't have 'predict' attribute!")

				ensemble_pred_matrix = sum(list_of_pred_matrices)
				ensemble_predictions = np.argmax(ensemble_pred_matrix, axis =1)
					
				ensemble_predictions = unique_classes[ensemble_predictions]
				ensemble_predictions = ensemble_predictions[:-len(unique_classes)]

				return ensemble_predictions



			def predict_proba(self, X):

				if isinstance(X, (pd.DataFrame,pd.Series)):
					X= X.values

				ensemble_prob_preds = []
				models_with_prob_attr = []

				for idx, model in enumerate(self.models):

					if hasattr(model, "predict_proba"):

						model_prob_preds = model.predict_proba(X)
							
						ensemble_prob_preds.append(model_prob_preds)

						models_with_prob_attr.append(idx)

					else:
						print("Model", model, "doesn't have a predict_proba method")

				prob_weights = self.model_weights[models_with_prob_attr]
				
				ensemble_prob_preds = np.array(ensemble_prob_preds).transpose((2,1,0))
				prob_weights = np.array(prob_weights).transpose()

				ensemble_probabilities = np.matmul(ensemble_prob_preds,prob_weights).transpose()
			
				print("If the probabilities are NaNs this may be because a constant has been added to the design matrix (predictors)")

				return ensemble_probabilities

			
			def decision_function(self, X):

				if isinstance(X, (pd.DataFrame,pd.Series)):
					X= X.values

				models_with_decfunc_attr = []
				ensemble_decision_functions = []

				for idx, model in enumerate(self.models):

					if hasattr(model, "decision_function"):

						model_decision_function = model.decision_function(X)

						ensemble_decision_functions.append(model_decision_function)

						models_with_decfunc_attr.append(idx)

					else:
						print("Model", model, "doesn't have a decision_function method")

				decision_weights = self.model_weights[models_with_decfunc_attr]

				ensemble_decision_functions = np.array(ensemble_decision_functions).transpose((2,1,0))
				decision_weights = np.array(decision_weights).transpose()

				ensemble_decision_function = np.matmul(ensemble_decision_functions, decision_weights).transpose()

				print("If the decisions are NaNs this may be because a constant has been added to the design matrix (predictors)")

				return ensemble_decision_function


			def score(self, X = "default", y = "default", Metric = "precision"):
				
				if isinstance(X, str):
					if X =="default":
						X = self.training_X
				
				if isinstance(y, str):
					if y =="default":
						y = self.training_X
				

				if isinstance(X, (pd.DataFrame,pd.Series)):
					X= X.values

				if Metric == "precision":
					return self.precision_score(X,y)
				
				elif Metric == "accuracy":
					return self.accuracy_score(X,y)
				
				elif Metric == "log_loss":
					return self.log_loss(X,y)


			def log_loss(self, X = "default", y = "default"):
				
				if isinstance(X, str):
					if X =="default":
						X = self.training_X
				
				if isinstance(y, str):
					if y =="default":
						y = self.training_y

				if isinstance(X, (pd.DataFrame,pd.Series)):
					X= X.values

				probabilities = self.predict_proba(X)

				loss = neg_log_loss(y_true = y , y_pred = probabilities, eps=1e-15, normalize=True, sample_weight=None,labels=None)

				return loss

			def precision_score(self,X = "default", y = "default"):
				
				if isinstance(X, str):
					if X =="default":
						X = self.training_X
				
				if isinstance(y, str):
					if y =="default":
						y = self.training_y

				if isinstance(X, (pd.DataFrame,pd.Series)):
					X= X.values

				predictions = self.predict(X)

				score = precision_score(y_true = y, y_pred = predictions, average="weighted")

				return score

			def accuracy_score(self,X = "default", y = "default"):
				
				if isinstance(X, str):
					if X =="default":
						X = self.training_X
				
				if isinstance(y, str):
					if y =="default":
						y = self.training_y

				if isinstance(X, (pd.DataFrame,pd.Series)):
					X= X.values

				predictions = self.predict(X)

				score = accuracy_score(y_true = y, y_pred = predictions)

				return score

		random_forest = Random_Forest_Model(Models = fully_fit_models , Scores = model_median_scores, X= X, y = y )

		return random_forest







			
		










