
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True) # float64

def _weighted_stats(A, w):
	""" Weighted mean/std over axis 0 """
	sum_weights = jnp.sum(w)
	avg = (w @ A) / sum_weights
	sd = jnp.sqrt(w @ (A - avg)**2 / sum_weights)
	return avg, jnp.where(sd > 1e-12, sd, 1.0)

def _soft_threshold(A, t):
	return jnp.sign(A) * jnp.maximum(jnp.abs(A) - t, 0.0)

def _fista(smooth, W0, b0, step, l1, n_iter):
	"""
	Accelerated proximal gradient (FISTA) with a fixed iteration count (fixed count --> vmap)

	Arguments:
		smooth: Differentiable part of the objective, smooth(W, b) -> scalar
		W0/b0: Initial weights/bias
		step: Step size
		l1: Strength of L1 penalty on W (never applied to b).
		n_iter: Number of iterations
	"""

	grad = jax.grad(smooth, argnums=(0, 1))

	def body(data, _):
		W, b, W_prev, b_prev, t = data

		t_next = (1 + jnp.sqrt(1 + 4*t**2)) / 2
		momentum = (t - 1) / t_next
		VW = W + momentum * (W - W_prev)
		vb = b + momentum * (b - b_prev)

		grad_W, grad_b = grad(VW, vb)

		W_new = _soft_threshold(VW - step * grad_W, step * l1)
		b_new = vb - step * grad_b

		return (W_new, b_new, W, b, t_next), None

	(W, b, *_), _ = jax.lax.scan(body, (W0, b0, W0, b0, 1.0), None, length=n_iter)
	return W, b

def fit_core(X, Y, w, lam, l1_ratio=0.0, task="regression", n_iter=3_000):
	"""
	Fits one regularized linear model.

	Arguments:
		X (jnp.array): features [n, p]
		Y (jnp.array): targets (one for classification) [n, k]
		w (jnp.array): sample weights, 0 excludes a sample
		lam (float): regularization strength
		l1_ratio (float): 0 = ridge, 1 = lasso --> lam * ( (1-l1_ratio)/2 * L2_norm + l1_ratio * L1_norm)
		task (str): "regression" or "classification"
		n_iter (int): number of FISTA iteractions

	Returns:
		params (dict): coeff [p, k] and bias [k] in feature units, coeff_std [p, k] in standardized units, y_scale [k]
	"""

	x_avg, x_sd = _weighted_stats(X, w)
	X_centered = (X - x_avg) / x_sd
	sum_weights = jnp.sum(w)
	p, k = X.shape[1], Y.shape[1]

	reg_strength_L2 = lam * (1 - l1_ratio)
	reg_strength_L1 = lam * l1_ratio

	if task == "regression":
		y_avg, y_sd = _weighted_stats(Y, w)
		Y_centered = (Y - y_avg) / y_sd # target centered, meaning optimal intercept is zero
		G = (X_centered.T * w) @ X_centered / sum_weights # weighted Gram matrix [p, p]
		C = (X_centered.T * w) @ Y_centered / sum_weights # weighted cross-covariance [p, k]

		if l1_ratio == 0.0: # closed-form ridge
			B = jnp.linalg.solve(G + reg_strength_L2 * jnp.eye(p), C)
		else: # iterative for non-pure-ridge

			def smooth(B, b): # squared loss written with G and G (dropped the constant)
				return 0.5 * jnp.sum(B * (G @ B)) - jnp.sum(B * C) + 0.5 * reg_strength_L2 * jnp.sum(B**2)
			step = 1 / (jnp.linalg.eigvalsh(G)[-1] + reg_strength_L2)
			B, _ = _fista(smooth, jnp.zeros((p, k)), jnp.zeros(k), step, reg_strength_L1, n_iter)

		coeff = B / x_sd[:, None] * y_sd
		bias = y_avg - x_avg @ coeff
		return {"coeff": coeff, "bias": bias, "coeff_std": B, "y_scale": y_sd}

	if task == "classification":
		def smooth(B, b):
			log_p = jax.nn.log_softmax(X_centered @ B + b, axis=1)
			return -jnp.sum(w[:, None] * Y * log_p) / sum_weights + 0.5 * reg_strength_L2 * jnp.sum(B**2)

		X1 = jnp.concatenate([X_centered, jnp.ones((X_centered.shape[0], 1))], axis=1) # intercept column
		step = 1 / (0.5 * jnp.linalg.eigvalsh((X1.T * w) @ X1 / sum_weights)[-1] + reg_strength_L2) # softmax Hessian <= 1/2
		B, b = _fista(smooth, jnp.zeros((p, k)), jnp.zeros(k), step, reg_strength_L1, n_iter)

		coeff = B / x_sd[:, None]
		bias  = b - x_avg @ coeff
		return {"coeff": coeff, "bias": bias, "coeff_std": B, "y_scale": jnp.ones(k)}

	raise ValueError(f"fit_core: task must be 'regression' or 'classification', received {task}")

def predict_core(params, X, task="regression"):
	""" Regression --> predicted targets.  Classification --> class probabilities. """
	out = X @ params["coeff"] + params["bias"]
	return jax.nn.softmax(out, axis=1) if task == "classification" else out

def fit_grid(X, Y, W, lams, L1_ratio=0.0, task="regression", n_iter=3_000):
	"""
	Fits every (split, lambda) combination at once.

	Arguments:
		W: one row of sample weights per split [s, n]
		lams: regularization strengths [l]

	Returns:
		params with leading axes (s, l), and predictions on ALL samples with shape (s, l, n, k)
	"""

	@jax.jit
	def run(X, Y, W, lams):
		fit = lambda w, lam: fit_core(X, Y, w, lam, L1_ratio, task, n_iter)
		params = jax.vmap(jax.vmap(fit, in_axes=(None, 0)), in_axes=(0, None))(W, lams)
		preds = jax.vmap(jax.vmap(lambda prm: predict_core(prm, X, task)))(params)
		return params, preds

	return run(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(W), jnp.asarray(lams))

class regularized_linear:
	"""
	add a description later

	Attributes:
		regularization (str | None): "ridge", "lasso", "elastic_net", or None (ordinary least squares)
		reg_strength (float): lambda regularization strength
		L1_ratio (float): only used for elastic_net, allows for both ridge/lasso
		task (str): "regression" or "classification"
	"""

	def __init__(
		self,
		regularization: str | None = "ridge",
		reg_strength: float = 0.1,
		L1_ratio: float = 0.5,
		task: str = "regression",
		n_iter: int = 3_000
	):
		ratios = {"ridge": 0.0, "lasso": 1.0, "elastic_net": L1_ratio, None: 0.0}
		if regularization not in ratios:
			raise ValueError(f"regularization must be one of {list(ratios)}, received: {regularization}")

		self.regularization = regularization
		self.reg_strength = 0.0 if regularization is None else reg_strength
		self.L1_ratio = ratios[regularization]
		self.task = task
		self.n_iter = n_iter
		self.params = None

	def fit(self, X, y, sample_weight=None):
		X = jnp.asarray(X)
		y = jnp.asarray(y)

		if self.task == "classification" and y.ndim == 1: # integer labels --> one-hot
			y = jax.nn.one_hot(y, int(y.max()) + 1)
		if y.ndim == 1:
			y = y[:, None]
		w = jnp.ones(X.shape[0]) if sample_weight is None else jnp.asarray(sample_weight)

		self.params = fit_core(X, y, w, self.reg_strength, self.L1_ratio, self.task, self.n_iter)
		return self

	@property
	def weights(self):
		return self.params["coeff"]

	@property
	def bias(self):
		return self.params["bias"]

	def predict(self, X):
		out = predict_core(self.params, jnp.asarray(X), self.task)
		return jnp.argmax(out, axis=1) if self.task == "classification" else out

	def predict_proba(self, X):
		return predict_core(self.params, jnp.asarray(X), "classification")

	def mse(self, X, y):
		y = jnp.asarray(y)
		return jnp.mean((self.predict(X) - (y[:, None] if y.ndim == 1 else y))**2)
