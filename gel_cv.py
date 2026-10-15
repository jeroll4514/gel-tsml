"""
Splitting always happens by experiment, never by window.  Thus windows from one experiment overlap
and are stronly autocorrelated, so a random window split leads the test experiment into training.

Every split here is a row of sample weights.  One fit_grid call fits all of them for all lambdas:
	rows 0 .. P-1 (test e, val v) pairs, train on remaining experiments -> train/val/test scheme
	rows P .. P+E-1 test e only, train on all others -> refits for nested leave-one-out

Two ways to read off a test score:
	"pairs": for each ordered (test, val) pair, pick lambda on val, score on test. E*(E-1) results
	"nested": for each test e, average the val curves over every v != e, pick lambda, refit on all experiments
		except for e, score on e.  E results, lower variance, uses all data
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp

from itertools import permutations

from jax_linear import fit_grid as fit_grid_jax
from gel_windows import window_set
from sklearn_linear import fit_grid as fit_grid_sklearn

FIT_BACKENDS = {"jax": fit_grid_jax, "sklearn": fit_grid_sklearn}

def balanced_weights(exp_idx: np.ndarray, include: np.ndarray) -> np.ndarray:
	""" Each included experiments gets total weight 1 regardless of its number of frames """
	counts = np.bincount(exp_idx, minlength=include.size)
	return np.where(include[exp_idx], 1/np.maximum(counts[exp_idx], 1), 0.0) 

def make_splits(exp_idx: np.ndarray, num_experiments: int, tests: list | None = None) -> tuple[np.ndarray, list, list]:
	"""
	Arguments:
		tests (list | None): only build the splits for these test experiments, None = every experiment

	Returns:
		W (splits, samples): training weights
		pairs: list of (test, val) for the first rows of W
		refits: list of test for the last rows of W
	"""

	present = np.unique(exp_idx) # experiments with at least one window
	tests = present if tests is None else [test for test in tests if test in present] # overlap of tests and present
	pairs = [(test, val) for test, val in permutations(present, 2) if test in tests]
	refits = list(tests)

	W = []
	for test, val in pairs:
		include = np.isin(np.arange(num_experiments), present) & ~np.isin(np.arange(num_experiments), [test, val])
		W.append(balanced_weights(exp_idx, include))
	for test in refits:
		include = np.isin(np.arange(num_experiments), present) & (np.arange(num_experiments) != test)
		W.append(balanced_weights(exp_idx, include))

	return np.asarray(W), pairs, refits

def per_sample_loss(preds, y, task, y_scale):
	"""
	Regression: squared error averaged over outputs
	Classification: cross-entropy.

	preds (..., n, k), y (n, k), y_scale (..., k) -> (..., n)
	"""
	if task == "regression":
		return jnp.mean(((preds - y) / y_scale[..., None, :])**2, axis=-1)
	return -jnp.sum(y * jnp.log(jnp.clip(preds, 1e-12, 1.0)), axis=-1)

def per_experiment_mean(values, exp_idx, num_experiments):
	""" (..., n) -> (..., num_experiments), mean over each experiment's samples """
	onehot = np.eye(num_experiments)[exp_idx]
	counts = np.maximum(onehot.sum(axis=0), 1)
	return values @ onehot / counts

def cross_validate(
	ws: window_set,
	target: str,
	task: str,
	lams: np.ndarray,
	L1_ratio: float = 0.0,
	baseline: np.ndarray | None = None,
	balance_classes: bool = False,
	n_iter: int = 3_000,
	backend: str = "jax",
	tests: list | None = None
) -> dict:
	"""
	Runs the full (split x lambda) grid for one target and summarizes both evaluation schemes.

	Arguments:
		ws: window_set from make_windows
		target: key for ws.targets
		task: "regression" or "classification"
		lams: lambda grid
		L1_ratio: 0 ridge, 1 lasso
		baseline (n, k) optional: baseline predictions (e.g. ws.persistence()) scored the same way
			Default: training-fold mean (regression) or training-fold class frequencies (classification)
		balance_classes: classification only, reweight so every class has equal total training weight.
			Needed for the outcome task: holding out one experimnet leaves its class under-represented, which
			otherwise biases every held-out prediction away from the true class.
		n_iter (int): the maximum number of steps to determine converged coefficients
		backend (str): "jax" or "sklearn", which fit_grid to use
		tests (list | None): only score these test experiments (used by cross_validate_foldwise), None = all
	
	Returns dict with:
		"pairs": DataFrame, one row per (test, val) pair
		"nested": DataFrame, one row per test experiment
		"val_curves": (num test experiments, num lambdas) inner-CV loss used to pick lambda (nested)
		"params"/"preds": the refit models (one per test experimnet) at their chosen lambda
	"""

	X, y = ws.X, ws.targets[target]
	num_experiments = ws.num_experiments
	W, pairs, refits = make_splits(ws.exp_idx, num_experiments, tests)
	if balance_classes and task == "classification":
		class_weight = W @ y # (splits, k)
		W = W * ((y / np.where(class_weight > 0, class_weight, np.inf)[:, None, :]).sum(axis=2))

	params, preds = FIT_BACKENDS[backend](X, y, W, lams, L1_ratio, task, n_iter)
	loss = per_sample_loss(preds, jnp.asarray(y), task, params["y_scale"]) # (splits, lams, n)
	exp_loss = np.asarray(per_experiment_mean(loss, ws.exp_idx, num_experiments)) # (splits, lams, num_experiments)

	# baselines, scored per split so they use the same normalization and training data as the model
	if task == "classification":
		freq = (W @ y) / W.sum(axis=1, keepdims=True) # (splits, k)
		base_loss = -np.log(np.clip(freq, 1e-12, 1)) @ y.T # (splits n)
	else:
		y_scale = np.asarray(params["y_scale"][:, 0]) # same for every lambda
		if baseline is None:
			baseline = ((W @ y) / W.sum(axis=1, keepdims=True))[:, None, :] # (splits, 1, k) training mean
		base_loss = np.mean(((baseline - y) / y_scale[:, None, :])**2, axis=-1)
	base_exp = per_experiment_mean(base_loss, ws.exp_idx, num_experiments)

	name = lambda e: str(ws.experiments[e])
	num_pairs = len(pairs)

	rows = []
	for s, (test, val) in enumerate(pairs):
		best = int(np.argmin(exp_loss[s, :, val]))
		rows.append({
			"test": name(test), "val": name(val), "lam": lams[best], "val_loss": exp_loss[s, best, val],
			"test_loss": exp_loss[s, best, test], "baseline_loss": base_exp[s, test]
		})
	pairs_df = pd.DataFrame(rows)

	rows, curves, best_idx = [], [], []
	for r, test in enumerate(refits):
		inner = [s for s, (te, _) in enumerate(pairs) if te == test]
		curve = np.mean([exp_loss[s, :, pairs[s][1]] for s in inner], axis=0)
		curves.append(curve)

		best = int(np.argmin(curve))
		best_idx.append(best)

		rows.append({
			"test": name(test), "lam": lams[best], "val_loss": curve[best],
			"test_loss": exp_loss[num_pairs + r, best, test], "baseline_loss": base_exp[num_pairs + r, test]
		})
	nested_df = pd.DataFrame(rows)

	edge = nested_df["lam"].isin([np.min(lams), np.max(lams)])
	if edge.any():
		print(f"cross_validate: WARNING, {target} ({L1_ratio=}) chose a lambda at the edge of the grid for "
			f"{edge.sum()}/{len(edge)} test experiments, consider widening lams.")

	refit_rows = num_pairs + np.arange(len(refits))
	chosen_params = {key: np.asarray(val[refit_rows, best_idx]) for key, val in params.items()}
	chosen_preds = np.asarray(preds[refit_rows, best_idx]) # (tests, n, k)

	return {
		"pairs": pairs_df, "nested": nested_df, "val_curves": np.asarray(curves), "lams": lams,
		"refits": refits, "params": chosen_params, "preds": chosen_preds
	}

def cross_validate_foldwise(
	build_windows,
	num_experiments: int,
	target: str,
	task: str,
	lams: np.ndarray,
	L1_ratio: float = 0.0,
	persistence_baseline: bool = False,
	balance_classes: bool = False,
	n_iter: int = 3_000,
	backend: str = "jax"
) -> dict:
	"""
	cross_validate for features that must be fit per fold (e.g. PCA): for each test experiment, build_windows(test)
	returns a window_set whose features were fit WITHOUT that experiment, and only that experiment's splits are run
	on it.  Same total work as one cross_validate call, same return dict.

	Arguments:
		build_windows: function(test expreiment index) -> window_set (same windows every time, only X differs)
		num_experiments: number of experiments
		persistence_baseline: score the baseline as ws.persistence() (the forecast task)
		other arguments: see cross_validate
	"""

	folds = []
	for test in range(num_experiments):
		ws = build_windows(test)
		if not np.any(ws.exp_idx == test):
			continue # experiment too short to give any windows
		baseline = ws.persistence() if persistence_baseline else None
		folds.append(cross_validate(
			ws, target, task, lams, L1_ratio, baseline, balance_classes, n_iter, backend, tests=[test]
		))
	
	return {
		"pairs": pd.concat([fold["pairs"] for fold in folds], ignore_index=True),
		"nested": pd.concat([fold["nested"] for fold in folds], ignore_index=True),
		"val_curves": np.concatenate([fold["val_curves"] for fold in folds]),
		"lams": lams,
		"refits": [test for fold in folds for test in fold["refits"]],
		"params": {key: np.concatenate([fold["params"][key] for fold in folds]) for key in folds[0]["params"]},
		"preds": np.concatenate([fold["preds"] for fold in folds])
	}

def test_predictions(ws: window_set, result: dict, target: str) -> pd.DataFrame:
	""" 
	Out-of-sample predictions of the nested refits: each experiment predicted by the model
	that never saw it
	"""

	rows = []
	for r, test in enumerate(result["refits"]):
		mask = ws.exp_idx == test
		y = ws.targets[target][mask]
		P = result["preds"][r][mask]
		for i, t in enumerate(ws.t[mask]):
			row = {"experiment": str(ws.experiments[test]), "t": t}
			row.update({f"true_{k}": y[i, k] for k in range(y.shape[1])})
			row.update({f"pred_{k}": P[i, k] for k in range(P.shape[1])})
			rows.append(row)

	return pd.DataFrame(rows)

def experiment_level_outcome(ws: window_set, result: dict, classes: tuple) -> pd.DataFrame:
	"""
	Combines window-level class probabilities into one call per held-out experiment (mean log-probability)
	and also reports how early in the experimnet the running call becomes correct.
	"""

	rows = []
	for r, test in enumerate(result["refits"]):
		mask = ws.exp_idx == test
		log_p = np.log(np.clip(result["preds"][r][mask], 1e-12, 1))
		truth = int(np.argmax(ws.targets["outcome"][mask][0]))
		running = np.argmax(np.cumsum(log_p, axis=0), axis=1) # decision using frames seem so far
		wrong = np.flatnonzero(running != truth)
		first_stable = 0 if wrong.size == 0 else wrong[-1] + 1 # first window after which it stays correct

		mean_log_p = log_p.mean(axis=0)
		undecided = np.ptp(mean_log_p) < 1e-3 # lasso can zero every coefficient -> uniform probabilities and argmax are meaningless

		rows.append({
			"experiment": str(ws.experiments[test]),
			"true": classes[truth],
			"predicted": "undecided" if undecided else classes[int(np.argmax(mean_log_p))],
			**{f"p_{c}": np.exp(mean_log_p)[k] for k, c in enumerate(classes)},
			"window_accuracy": np.nan if undecided else np.mean(np.argmax(log_p, axis=1) == truth),
			"correct_from_frame": ws.t[mask][first_stable] if first_stable < mask.sum() and not undecided else np.nan
		})

	return pd.DataFrame(rows)
