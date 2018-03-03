import numpy as np
import pandas as pd
import cvxopt
import math
import time
from sklearn.svm import SVC

# def Gauss_Kernel(x1,x2, sigma =2):
# 	delta_x = x1-x2
# 	delta_x = delta_x.reshape((-1,1))
# 	return math.exp( -1*(np.dot(delta_x.transpose(), delta_x)) / 2*sigma**2 )

# def Linear_Kernel(x1, x2):
# 	x1, x2 = x1.reshape((-1,1)), x2.reshape((-1,1))
# 	return np.dot(x1.transpose(), x2)

# def Polynomial_Kernel(x1, x2, p=3):
# 	x1, x2 = x1.reshape((-1,1)), x2.reshape((-1,1))
# 	return (1 + np.dot(x1.transpose(), x2))**p


def Gauss_Kernel(X1, X2, gamma = 1):
	tens = [0 for i in range(len(X1))]
	for i in range(len(X1)):
		tens[i] = (X2-X1[i]).transpose()
   
	tens = np.array(tens).transpose((0,2,1))
	K= (tens * tens).sum(axis=2)
	
	def Gauss_each_element(A, gamma_ = gamma):
		
		B = -1*(A * gamma_)
		
		return np.exp(B)
	
	K= Gauss_each_element(K)
	
	return K

def Linear_Kernel(X1,X2):
	K = np.dot(X1, X2.transpose())
	return K

def Polynomial_Kernel(X1, X2, p =3):
	K = np.power(np.dot(X1, X2.transpose()) + 1.0, p)
	return K

encoder = np.vectorize(lambda y: 1 if y >0.1 else -1)


class My_SVM(SVC):
	"""
	Support Vector Classifier Object
	Takes Kernel, C, and tol parameters
	Attributes:
		support_vectors
		
	Methods:
		fit
		fit_predict
		predict

	"""

	def __init__(self, kernel = Linear_Kernel, C = 1, tol = 0.1, bias = True, encode = True, gamma = 2 ):
		
		if kernel == "linear":
			self.kernel = Linear_Kernel
		elif kernel == "gaussian":
			self.kernel = Gauss_Kernel
		elif kernel == "polynomial":
			self.kernel = Polynomial_Kernel
		else:
			self.kernel = kernel

		self.encode = encode
		self.C = C
		self.tol = tol
		self.bias = bias
		self.gamma = gamma




	def fit(self, X, y):
		
		if isinstance(X, pd.DataFrame):
			X = X.values
		if isinstance(y, (pd.Series, pd.DataFrame)):
			y = y.values    

		if self.encode:
			y_unique, y_inverse = np.unique(y, return_inverse = True)
			y= encoder(y_inverse)


		n_obs = X.shape[0]
		n_features = X.shape[1]
		self.n_features = n_features

		K = self.kernel(X, X)

		y= y.reshape((-1,1))

		tmp1 = np.dot(y,y.transpose())
		P = np.multiply(tmp1, K)

		P = cvxopt.matrix(P)

		q = -1 * np.ones(n_obs)
		q = cvxopt.matrix(q)

		tmp2 = np.diag(-1*np.ones(n_obs))
		tmp3 = np.diag(np.ones(n_obs))
		G = np.vstack([tmp2,tmp3])
		G = cvxopt.matrix(G)

		h = np.zeros(n_obs)
		if self.C > 0:
			tmp4 = np.full(shape = n_obs, fill_value = self.C)
			h = np.hstack([h, tmp4])
		h = cvxopt.matrix(h)

		A =cvxopt.matrix(y.transpose().astype(float))

		b = np.array([0.0])
		b = cvxopt.matrix(b)



		sol = cvxopt.solvers.qp(P,q,G,h,A,b)

		alpha = np.array(sol["x"]).reshape((-1,))
		
		obs_idx = np.arange(n_obs)

		sv_mask = alpha > 0.05

		support_vectors_idx = obs_idx[sv_mask]

		self.alpha = alpha[sv_mask]

		self.support_vectors = X[support_vectors_idx,:]
		
		self.support_vector_labels = y[support_vectors_idx]

		self.num_support_vectors = len(support_vectors_idx)

		if self.bias ==True:
			#  ## Calculate the bias needed
			# Need the correct support vectors first
			if self.C > 0:
				exact_alpha_mask = (alpha > 0.05) & (alpha < self.C )	
				exact_alpha = alpha[exact_alpha_mask]
				exact_sv_idx = obs_idx[exact_alpha_mask]
				exact_sv = X[exact_sv_idx,:]
				
				exact_sv_labels = y[exact_sv_idx]
			
			else:
				exact_alpha = self.alpha
				exact_sv_idx = self.support_vectors
				exact_sv = self.support_vectors
				exact_sv_labels = self.support_vector_labels
			
			
			weighting = (exact_alpha * exact_sv_labels.reshape(-1,)).reshape((1,-1))
			# print("weight",weighting.shape)
			sv_kernel = self.kernel(exact_sv, exact_sv)
			# print("kernel",sv_kernel.shape)
			b_hat_array = exact_sv_labels.reshape((-1,)) - np.dot(weighting, sv_kernel)
			# print("b_hat",b_hat_array.shape)

			self.bias = b_hat_array.sum()/sum(exact_alpha_mask)
		
		elif isinstance(self.bias, (float,int)):
			pass

		else:
			self.bias = 0.0
		
		return self

		###


	def predict(self, X):
		#Correct data format
		if isinstance(X, pd.DataFrame):
			X = X.values
		#Initialise kernel matrix
		new_n_obs = X.shape[0]
		
		kernelized_test = self.kernel(self.support_vectors,X)

		tmp_arr = np.multiply(self.alpha, self.support_vector_labels.reshape((-1,))).transpose()
		tmp_arr = tmp_arr.reshape((-1,1)).transpose()
		
		self.decision_values = np.dot(tmp_arr ,kernelized_test) + self.bias
		print(self.bias)
		
		prediction_array = np.sign(self.decision_values).reshape((-1,))
		
		return prediction_array

		###


	def fit_predict(self, X, y):
		
		self.fit(X,y)
		
		return self.predict(X)

		###


	def decision_function(self, X):

		#Correct data format
		if isinstance(X, pd.DataFrame):
			X = X.values
		#Initialise kernel matrix
		new_n_obs = X.shape[0]
		
		kernelized_test = self.kernel(self.support_vectors,X)

		tmp_arr = np.multiply(self.alpha, self.support_vector_labels.reshape((-1,))).transpose()
		tmp_arr = tmp_arr.reshape((-1,1)).transpose()
		
		self.decision_values = np.dot(tmp_arr ,kernelized_test).reshape((-1,)) + self.bias

		return self.decision_values


	def score(self, X, y):
		
		preds = self.predict(X)

		preds = preds.reshape((-1,))
		y= y.reshape((-1,))

		score = sum(preds == y)/len(y)

		return score


















