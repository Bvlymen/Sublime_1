#Gradient Descent

class Lin_Reg_Gradient_Descent(object):

	def __init__(cost_func = "MSE", rate = 1, tol = 10):
		self.cost_func = cost_func
		self.rate = rate

		if cost_func =="MSE"

		self.descender = MSE


	def MSE(X,y, model):
		"""
		Beta_0 Grad = (f(x) - y)
		(Beta_i | i ≠ 0) Grad  = (f(x) - y)x_i 
		"""
		preds = model.predict(X)
		
		start_params = np.array(model.params)
		new_params = np.array(model.params)

		initial_mse = np.mean(np.power(preds - y,2))
		new_mse = np.mean(np.power(preds - y,2))
		

		delta_cost = tol + 1
		
		while delta_cost < tol:
			
			# new_params = np.array_like(params)

			grad_0 = np.mean(preds -y)

			grads_i = np.mean((preds - y) * X, axis = 1)

			full_grads = np.concat([grad_0, grads_i])

			new_params = new_params - self.rate * full_grads

			model.params = new_params

			new_preds = model.predict(X)

			delta_cost = new_mse - np.mean(np.power(new_preds - y,2))

			new_mse = np.mean(np.power(new_preds - y,2))

			print(delta_cost)

		final_params = new_params



