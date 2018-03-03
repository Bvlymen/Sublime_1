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

def Random_Forest_Classifier(X, y, Classifiers = [LDA, QDA, KNN, GNB, TREE], cv = 3, Bootstrap_Samples = 5, Rand_Param_Search = True, Rand_Search_Iterations = 5, Param_Dicts = "default", Random_State = None):
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
	for i in range(Bootstrap_Samples):
		
		resample = np.random.choice(a= bootstrap_indices, replace = True, size = len(X))
		bootstrap_X = X[resample]
		bootstrap_y = y[resample]
		bootstrap_samples.append((bootstrap_X, bootstrap_y))

	if Param_Dicts == "guessed":
		
		GNB_dict = {"priors":[None]}
		LDA_dict = {"n_components":[None, 3, 5, 10,20,30,50]}
		QDA_dict = {"reg_param":np.linspace(0,1,8)}
		TREE_dict = {"criterion":["entropy", "gini"], "max_depth":[3,6,9,10,12], "min_samples_split":[10,20,30]}
		KNN_dict = {"n_neighbors":[5,10,20,35], "weights":["distance", "uniform"]}

		param_dicts = [LDA_dict, QDA_dict, KNN_dict, GNB_dict, TREE_dict]

	elif Param_Dicts == "default":

		param_dicts = np.full((len(Classifiers),), {})
	
	else:
		param_dicts = Param_Dicts

	#initialise list of fully fit and tuned models
	fully_fit_models = []
	model_median_scores = []
	#For loop - for each classifier do things
	for estimator in Classifiers:
		#Find estimator name
		estimator_name = str(estimator()).split("(")[0]
		
		#Initialise the scores and parameters list
		exec(estimator_name + "_scores = []")
		exec(estimator_name + "_params = []" )

		#For loop - for each bootstrap calculate hyperparameters and scores
		for idx, (bootstrap, param_dict) in enumerate(zip(bootstrap_samples, param_dicts)):
			#Check whether we Randomse or grid search the hyperparameters
			if (Rand_Param_Search == True) & (Rand_Search_Iterations <= len(np.array(list(param_dict.values())).ravel())):
				try:
					#Fit the model to a Randomised search CV
					hyper_search_cv = RandomizedSearchCV(estimator = estimator(), param_distributions = param_dict, cv = cv, n_iter = Rand_Search_Iterations, iid = False, random_state = Random_State)
				
				except ValueError:
					try:
						cont
						if (cont == "y") | (cont == "yes"):
							hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = cv, iid = False)
					
					except NameError:
						print("Most likely Paramter Grid too small for n_iter in RandomizedSearchCV.")
						cont = str(input("Can try to continue with GridSearchCV? [y/n]:"))
				
						if (cont == "y") | (cont == "yes"):
							
							hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = cv, iid = False)
						
						else:
							print("Stopping")

			elif (Rand_Param_Search == True) & (Rand_Search_Iterations >= len(np.array(list(param_dict.values())).ravel())):
				try:
					cont
					if (cont == "y") | (cont == "yes"):
						hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = cv, iid = False)
					
				except NameError:

					print("Most likely Paramter Grid too small for n_iter in RandomizedSearchCV.")
					cont = str(input("Can try to continue with GridSearchCV? [y/n]:"))
			
					if (cont == "y") | (cont == "yes"):
						hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = cv, iid = False)
					else:
						print("Stopping")
						break
			
			elif Rand_Param_Search == False:
				#Fit the model to a Grid search CV
				hyper_search_cv = GridSearchCV(estimator = estimator(), param_grid = param_dict, cv = cv, iid = False)
					
				
			#For invalid argument values
			else:

				raise ValueError("Invalid argument value. Should be of type bool.")

			#Fit the model and search for hyperparams
			
			hyper_search_cv.fit(X, y.ravel())
			

			best_model_params = hyper_search_cv.best_params_
			model_score = hyper_search_cv.best_score_

			exec(estimator_name + "_scores.append(model_score)")
			# exec(estimator_name + "_params.append(tuple(best_model_params.items()))")

			print("Completed Fitting of", estimator_name, "for bootstrap:", idx)
			
			if idx ==Bootstrap_Samples-1:
				print("Scores for", estimator_name+":",eval(estimator_name+"_scores"))

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
		print("Added a classifier:", estimator_name, "to the ensemble!")

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


		def score(self, X = "default", y = "default"):
			
			if isinstance(X, str):
				if X =="default":
					X = self.training_X
			
			if isinstance(y, str):
				if y =="default":
					y = self.training_X
			

			if isinstance(X, (pd.DataFrame,pd.Series)):
				X= X.values

			return self.log_loss(X,y)


		def log_loss(self, X = "default", y = "default"):
			
			if isinstance(X, str):
				if X =="default":
					X = self.training_X
			
			if isinstance(y, str):
				if y =="default":
					y = self.training_X

			if isinstance(X, (pd.DataFrame,pd.Series)):
				X= X.values

			probabilities = self.predict_proba(X)
			

			def compute_log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None,
             labels=None):
			    """Log loss, aka logistic loss or cross-entropy loss.
			    This is the loss function used in (multinomial) logistic regression
			    and extensions of it such as neural networks, defined as the negative
			    log-likelihood of the true labels given a probabilistic classifier's
			    predictions. The log loss is only defined for two or more labels.
			    For a single sample with true label yt in {0,1} and
			    estimated probability yp that yt = 1, the log loss is
			        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
			    Read more in the :ref:`User Guide <log_loss>`.
			    Parameters
			    ----------
			    y_true : array-like or label indicator matrix
			        Ground truth (correct) labels for n_samples samples.
			    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
			        Predicted probabilities, as returned by a classifier's
			        predict_proba method. If ``y_pred.shape = (n_samples,)``
			        the probabilities provided are assumed to be that of the
			        positive class. The labels in ``y_pred`` are assumed to be
			        ordered alphabetically, as done by
			        :class:`preprocessing.LabelBinarizer`.
			    eps : float
			        Log loss is undefined for p=0 or p=1, so probabilities are
			        clipped to max(eps, min(1 - eps, p)).
			    normalize : bool, optional (default=True)
			        If true, return the mean loss per sample.
			        Otherwise, return the sum of the per-sample losses.
			    sample_weight : array-like of shape = [n_samples], optional
			        Sample weights.
			    labels : array-like, optional (default=None)
			        If not provided, labels will be inferred from y_true. If ``labels``
			        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
			        assumed to be binary and are inferred from ``y_true``.
			        .. versionadded:: 0.18
			    Returns
			    -------
			    loss : float
			    Examples
			    --------
			    >>> log_loss(["spam", "ham", "ham", "spam"],  # doctest: +ELLIPSIS
			    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
			    0.21616...
			    References
			    ----------
			    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
			    p. 209.
			    Notes
			    -----
			    The logarithm used is the natural logarithm (base-e).
			    """

			    def _weighted_sum(sample_score, sample_weight, normalize=False):
				    if normalize:
				        return np.average(sample_score, weights=sample_weight)
				    elif sample_weight is not None:
				        return np.dot(sample_score, sample_weight)
				    else:
				        return sample_score.sum()

				        
			    y_pred = check_array(y_pred, ensure_2d=False)
			    check_consistent_length(y_pred, y_true)

			    lb = LabelBinarizer()

			    if labels is not None:
			        lb.fit(labels)
			    else:
			        lb.fit(y_true)

			    if len(lb.classes_) == 1:
			        if labels is None:
			            raise ValueError('y_true contains only one label ({0}). Please '
			                             'provide the true labels explicitly through the '
			                             'labels argument.'.format(lb.classes_[0]))
			        else:
			            raise ValueError('The labels array needs to contain at least two '
			                             'labels for log_loss, '
			                             'got {0}.'.format(lb.classes_))

			    transformed_labels = lb.transform(y_true)

			    if transformed_labels.shape[1] == 1:
			        transformed_labels = np.append(1 - transformed_labels,
			                                       transformed_labels, axis=1)

			    # Clipping
			    y_pred = np.clip(y_pred, eps, 1 - eps)

			    # If y_pred is of single dimension, assume y_true to be binary
			    # and then check.
			    if y_pred.ndim == 1:
			        y_pred = y_pred[:, np.newaxis]
			    if y_pred.shape[1] == 1:
			        y_pred = np.append(1 - y_pred, y_pred, axis=1)

			    # Check if dimensions are consistent.
			    transformed_labels = check_array(transformed_labels)
			    if len(lb.classes_) != y_pred.shape[1]:
			        if labels is None:
			            raise ValueError("y_true and y_pred contain different number of "
			                             "classes {0}, {1}. Please provide the true "
			                             "labels explicitly through the labels argument. "
			                             "Classes found in "
			                             "y_true: {2}".format(transformed_labels.shape[1],
			                                                  y_pred.shape[1],
			                                                  lb.classes_))
			        else:
			            raise ValueError('The number of classes in labels is different '
			                             'from that in y_pred. Classes found in '
			                             'labels: {0}'.format(lb.classes_))

			    # Renormalize
			    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
			    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

			    return _weighted_sum(loss, sample_weight, normalize)

			loss = compute_log_loss(y_true = y , y_pred = probabilities, eps=1e-15, normalize=True, sample_weight=None,labels=None)

			return loss

	random_forest = Random_Forest_Model(Models = fully_fit_models , Scores = model_median_scores, X= X, y = y )


	return random_forest







			
		










