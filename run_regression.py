
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_auc_score

from gel_ts import time_series
from gel_windows import OUTCOMES, detect_events, make_windows
from gel_cv import cross_validate, cross_validate_foldwise, test_predictions, experiment_level_outcome
from spatial_features import normalize_per_experiment

# TEMP since I use WSL
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# ============================================================= #
# Settings

NPZ_PATH = Path("Maskeddatatoupload.npz")
FEATURES = ("u_px", "v_px", "exx", "eyy", "exy", "I1", "Henky", "E_max", "E_min")

SPATIAL = "pca"         # "average" = spatial averaging, "summary" = mean + the statistics below, "pca" = summary + PCA scores
AGGREGATIONS = ("mean", "std", "p99") # see spatial_features.AGGREGATIONS
DERIVED = ("disp",)     # see spatial_features.DERIVED
NORMALIZE_TASKS = ("outcome",) # tasks that see per-gel robust z-scores (helps outcome, hurts event_soon)

PCA_FIELDS = ("E_max", "I1", "Henky", "disp") # "pca" only, one PCA per field on a GRID_SIZE x GRID_SIZE grid
N_COMPONENTS = 3    # "pca" only, components kept per field
GRID_SIZE = 12      # "pca" only, cells per side of the common binning grid
PCA_FOLDWISE = True # "pca" only, refit the PCA without each test experiment (False = fit once on everything, leaks) 

LOOKBACK = 4            # frames of history per sample (training_length / frame_interval)
HORIZON = 3             # frames ahead (prediction_length / frame_interval)
EVENT_FEATURE = "E_max" # spatially avearged signal that defines a contraction
EVENT_THRESHOLD = 0.1   # E_max above this = contracting, rising edge = event onset
EVENT_FRAMES = None     # or {"20240824-2-rip-y": [8, 12], ...} to overide the detector with hand labels
LAMS = np.logspace(-3, 3, 31) # regularization strengths
MODELS = {"ridge": 0.0, "lasso": 1.0} # name --> L1_ratio
BACKEND = "jax"         # "jax" or "sklearn" (same results; testing shows sklearn lasso is much slower)
OUT_DIR = Path("results")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
OUT_DIR.mkdir(exist_ok=True)

# ============================================================= #

ts = time_series(
	file_path=NPZ_PATH, features=FEATURES, spatial=SPATIAL, aggregations=AGGREGATIONS, derived=DERIVED,
	pca_fields=PCA_FIELDS, n_components=N_COMPONENTS, grid_size=GRID_SIZE
)
features = ts.features # expanded names in "summary"/"pca" mode
onsets = detect_events(ts.X_full, features, EVENT_FEATURE, EVENT_THRESHOLD, EVENT_FRAMES, ts.experiments)
ws = make_windows(ts.X_full, ts.experiments, features, LOOKBACK, HORIZON, onsets=onsets)
ws_norm = make_windows(normalize_per_experiment(ts.X_full), ts.experiments, features, LOOKBACK, HORIZON, onsets=onsets) # same windows, normalized X

FOLDWISE = SPATIAL == "pca" and PCA_FOLDWISE

fold_cache = {}
def fold_windows(test: int, normalize: bool):
	""" Windows whose PCA was fit on every experiment except test (cached: each fold is reused by every task) """
	if (test, normalize) not in fold_cache:
		X = ts.pca_X(fit_on=[e for e in range(len(ts.experiments)) if e != test])
		X = normalize_per_experiment(X) if normalize else X
		fold_cache[test, normalize] = make_windows(X, ts.experiments, features, LOOKBACK, HORIZON, onsets=onsets)
	return fold_cache[test, normalize]

print(f"{ts.X_full.shape = }, {ws.X.shape = }")
for e, name in enumerate(ts.experiments):
	print(f"  {name:28s} windows={np.sum(ws.exp_idx == e):3d}  event onsets at frames {np.flatnonzero(onsets[e].tolist())}")

# ========== Tasks ========== #
#  name  target  task  baseline  balance_classes
TASKS = [
	("forecast", "forecast", "regression", "persistence", False), # features at t+H vs. "nothing changes"
	("time_to_event", "time_to_event", "regression", None, False), # frames until next onset
	("event_soon", "event_soon", "classification", None, False), # onset within H frames?
	("outcome", "outcome", "classification", None, True) # tear / detach / none
]

summary, results = [], {}
for task_name, target, task, baseline, balance in TASKS:
	for model, L1_ratio in MODELS.items():
		start = time.perf_counter()

		normalize = task_name in NORMALIZE_TASKS
		if FOLDWISE:
			res = cross_validate_foldwise(
				lambda test: fold_windows(test, normalize), len(ts.experiments), target, task, LAMS, L1_ratio,
				baseline == "persistence", balance, backend=BACKEND
			)
		else:
			ws_task = ws_norm if normalize else ws
			res = cross_validate(
				ws_task, target, task, LAMS, L1_ratio, ws_task.persistence() if baseline == "persistence" else None,
				balance, backend=BACKEND
			)
		results[task_name, model] = res

		nested, pairs = res["nested"], res["pairs"]
		summary.append({
			"task": task_name, "model": model, "nested_test_loss": nested["test_loss"].mean(),
			"nested_test_std": nested["test_loss"].std(), "pairs_test_loss": pairs["test_loss"].mean(),
			"pairs_test_std": pairs["test_loss"].std(), "baseline_loss": nested["baseline_loss"].mean(),
			"median_lam": nested["lam"].median(), "seconds": time.perf_counter() - start
		})

		nested.to_csv(OUT_DIR / f"{task_name}_{model}_nested.csv", index=False)
		pairs.to_csv(OUT_DIR / f"{task_name}_{model}_pairs.csv", index=False)
		test_predictions(ws, res, target).to_csv(OUT_DIR / f"{task_name}_{model}_predictions.csv", index=False)

summary = pd.DataFrame(summary)
print("\nLoss on held-out experiments (lower is better; regression in units of target std, classification = cross-entropy)")
print(summary.round(4).to_string(index=False))
summary.to_csv(OUT_DIR / "summary.csv", index=False)

for model in MODELS:
	preds = test_predictions(ws, results["event_soon", model], "event_soon")
	print(f"event_soon {model}: pooled held-out ROC AUC = {
	   roc_auc_score(preds['true_1'], preds['pred_1']):.3f}")

for model in MODELS:
	print(f"\nOutcome classification per held-out experiment ({model}):")
	print(experiment_level_outcome(ws, results["outcome", model], OUTCOMES).round(3).to_string(index=False))

# ========== Makes plots ========== #

fig, axs = plt.subplots(nrows=1, ncols=len(TASKS), figsize=(4 * len(TASKS), 3.5), constrained_layout=True)
for ax, (task_name, *_) in zip(axs, TASKS):
	for model in MODELS:
		curves = results[task_name, model]["val_curves"]
		ax.semilogx(LAMS, curves.mean(axis=0), label=model)
		ax.fill_between(LAMS, curves.min(axis=0), curves.max(axis=0), alpha=0.2)
	ax.set_title(task_name)
	ax.set_xlabel(r"$\lambda$")
axs[0].set_ylabel("inner-CV validation loss")
axs[0].legend()
fig.savefig(OUT_DIR / "validation_curves.png", dpi=150)

# Standardized coefficients (lag x feature) of each nested refit, averaged: which features/lags matter
for task_name in ("time_to_event", "event_soon"):
	coeff = results[task_name, "lasso"]["params"]["coeff_std"] # (tests, lookback*features, outputs)
	coeff = coeff[..., -1].mean(axis=0).reshape(LOOKBACK, len(features))

	fig, ax = plt.subplots(figsize=(max(7, 0.25*len(features)), 3), constrained_layout=True)

	lim = np.abs(coeff).max() or 1
	im = ax.imshow(coeff, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
	ax.set_xticks(range(len(features)), features, rotation=90, fontsize=6)
	ax.set_yticks(range(LOOKBACK), [f"t-{LOOKBACK - 1 - i}" for i in range(LOOKBACK)])
	ax.set_title(f"lasso coefficients, {task_name} (mean over held-out)")
	fig.colorbar(im)
	fig.savefig(OUT_DIR / f"coefficients_{task_name}.png", dpi=150)

# Held-out event-probability traces with the true onsets
preds = test_predictions(ws, results["event_soon", "lasso"], "event_soon")
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8), sharey=True, constrained_layout=True)

for ax, (name, df) in zip(axs.ravel(), preds.groupby("experiment")):
	e = list(ts.experiments).index(name)
	ax.plot(df["t"], df["pred_1"], label=f"P(onset within {HORIZON} frames)")
	ax.step(df["t"], df["true_1"], where="post", color="k", alpha=0.4, label="truth")
	for onset in np.flatnonzero(onsets[e]):
		ax.axvline(onset, color="r", ls=":", lw=1)
	ax.set_title(name, fontsize=9)

axs[0, 0].legend(fontsize=7)
fig.savefig(OUT_DIR / "event_soon_traces.png", dpi=150)

# PCA component maps (fit on every experiment), one row per field
if SPATIAL == "pca":
	ts.pca_X() # refit on all experiments for display
	pca = ts.pca
	fig, axs = plt.subplots(
		nrows=len(PCA_FIELDS), ncols=N_COMPONENTS, figsize=(2.6 * N_COMPONENTS, 2.4 * len(PCA_FIELDS)),
		constrained_layout=True, squeeze=False
	)
	for f, field in enumerate(PCA_FIELDS):
		for k in range(N_COMPONENTS):
			loading = pca["components"][f, k].reshape(GRID_SIZE, GRID_SIZE).T # rows = Y, cols = X for imshow
			lim = np.abs(loading).max() or 1
			axs[f, k].imshow(loading, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="upper")
			axs[f, k].set_title(f"{field} pc{k + 1} ({100 * pca["explained"][f, k]:.0f}%)", fontsize=9)
			axs[f, k].set_xticks([])
			axs[f, k].set_yticks([])
	fig.savefig(OUT_DIR / "pca_components.png", dpi=150)

print(f"\nWrote results to {OUT_DIR.resolve()}")
