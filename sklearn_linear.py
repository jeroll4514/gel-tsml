"""
sklearn drop-in for jax_linear.fit_grid, so gel_cv.cross_validate works unchanged.

Lambda conversion:
		Lasso/ElasticNet: alpha = lam          (sklearn rescales sample weights to mean 1)
		Ridge:            alpha = lam * sum(w) (sklearn uses a weighted SUM of squared errors)
		Logistic:         C = 1/(lam * sum(w)) (sklearn puts C on the summed loss instead of on the penalty)
"""

import warnings
import numpy as np

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression

def _weighted_stats(A, w):
	avg = (w @ A) / w.sum()
	sd = np.sqrt((w @ (A - avg)**2) / w.sum())
	return avg, np.where(sd > 1e-12, sd, 1.0)

def _softmax(A):
	A = A - A.max(axis=1, keepdims=True)
	return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)

def fit_grid(X, y, W, lams, L1_ratio=0.0, task="regression", n_iter=3_000):
	"""
	Fits every (split, lambda) combination.  Same signature/returns as jax_linear.fit_grid:
		params with leading axes (s, l), and predictions on ALL samples with shape (s, l, n, k)
	"""

	X, y, W, lams = map(np.asarray, (X, y, W, lams))
	S, L, (n, p), k = len(W), len(lams), X.shape, y.shape[1]
	order = np.argsort(lams)[::-1] # large -> small lambda, so warm starts follow regularization path

	param_tuple = (("coeff", (p, k)), ("bias", (k,)), ("coeff_std", (p, k)), ("y_scale", (k,)))
	params = {key: np.zeros((S, L) + shape) for key, shape in param_tuple}
	preds = np.zeros((S, L, n, k))

	for s, w_all in enumerate(W):
		train = w_all > 0
		w = w_all[train]
		x_avg, x_sd = _weighted_stats(X[train], w)
		X_centered = (X - x_avg) / x_sd

		if task == "regression":
			y_avg, y_sd = _weighted_stats(y[train], w)
			y_centered = (y - y_avg) / y_sd
			if L1_ratio == 0.0:
				model = Ridge(fit_intercept=False)
			elif L1_ratio == 1.0:
				model = Lasso(fit_intercept=False, warm_start=True, max_iter=n_iter) # sklearn default tol = 1e-4
			else:
				model = ElasticNet(l1_ratio=L1_ratio, fit_intercept=False, warm_start=True, max_iter=n_iter)
		elif task == "classification":
			labels = y.argmax(axis=1)
			solver = "lbfgs" if L1_ratio == 0.0 else "saga"
			model = LogisticRegression(l1_ratio=L1_ratio, solver=solver, warm_start=True, max_iter=n_iter, tol=1e-6)
		else:
			raise ValueError(f"fit_grid: task must be 'regression' or 'classification', received {task}.")

		for l in order:
			lam = lams[l]
			with warnings.catch_warnings():
				warnings.simplefilter("ignore") # ConvergenceWarning at tiny lambda, FutureWarning about penalty

				if task == "regression":
					model.set_params(alpha = lam * w.sum() if L1_ratio == 0.0 else lam)
					model.fit(X_centered[train], y_centered[train], sample_weight=w)

					B = model.coef_.reshape(k, p).T
					coeff = B / x_sd[:, None] * y_sd
					bias = y_avg - x_avg @ coeff

					params["y_scale"][s, l] = y_sd
					preds[s, l] = X @ coeff + bias

				else: # classification
					model.set_params(C = 1/(lam * w.sum()))
					model.fit(X_centered[train], labels[train], sample_weight=w)

					B, b = np.zeros((p, k)), np.zeros(k)
					if k == 2: # sklearn fits ONE weight vector for binary problems --> the logit of class 1
						B[:, 1], b[1] = model.coef_[0], model.intercept_[0]
					else:
						B[:, model.classes_], b[model.classes_] = model.coef_.T, model.intercept_

					coeff = B / x_sd[:, None]
					bias = b - x_avg @ coeff
					params["y_scale"][s, l] = 1.0
					preds[s, l] = _softmax(X @ coeff + bias)

			params["coeff"][s, l], params["bias"][s, l], params["coeff_std"][s, l] = coeff, bias, B

	return params, preds
