"""
Per-frame spatial summaries: instead of only the spatial mean of each field, keep the parts of the distribution
that carry localized activity.  Contraction precursors show up in the 95th/99th percentiles 1-2 frames before
they show up in the mean.
"""

import numpy as np
import pandas as pd

# name --> function applied to each (experiment, frame) group of one column
AGGREGATIONS = {
	"mean": "mean",
	"std": "std",
	"absmean": lambda s: s.abs().mean(),
	"p01": lambda s: s.quantile(0.01),
	"p05": lambda s: s.quantile(0.05),
	"p50": "median",
	"p95": lambda s: s.quantile(0.95),
	"p99": lambda s: s.quantile(0.99)
}

# per-point fields
DERIVED = {
	"disp": lambda d: np.hypot(d["u_px"], d["v_px"]) # displacement magnitude
}

def feature_name(feature: str, agg: str) -> str:
	""" Mean keeps the plain name, so EVENT_FEATURE = "E_max" still works, but others get a suffix """
	return feature if agg == "mean" else f"{feature}_{agg}"

def summarize_frames(
	data: dict,
	features: tuple | list,
	aggregations: tuple | list = ("mean", "std", "p99"),
	derived: tuple | list = ("disp",)
) -> tuple[dict, tuple]:
	"""
	Spatially summarizes every feature per experiment per frame.

	Arguments:
		data (dict): raw data from load_data (one row per grid point per frame)
		features (tuple | list): raw feature keys to summarize
		aggregations (tuple | list): keys of AGGREGATIONS
		derived (tuple | list): keys of DERIVED to compute and summarize as well

	Returns:
		output (dict): {"Name", "Frame", <summary feature>": array}, one row per experiment/frame
		names (tuple): summary feature names, in col order
	"""

	for agg in aggregations:
		if agg not in AGGREGATIONS:
			raise ValueError(f"summarize_frames: {agg=} must be one of {list(AGGREGATIONS)}")
	
	columns = list(features) + [d for d in derived if d not in features]

	df = pd.DataFrame({feature: np.asarray(data[feature], dtype=float) for feature in features})
	for d in derived:
		df[d] = DERIVED[d](data)
	df["Name"] = np.asarray(data["Name"]).astype(str)
	df["Frame"] = np.asarray(data["Frame"]).astype(int)

	grouped = df.groupby(["Name", "Frame"], sort=True)[columns]
	summary = pd.concat({agg: grouped.agg(AGGREGATIONS[agg]) for agg in aggregations}, axis=1) # cols (agg, feature)

	names = tuple(feature_name(feature, agg) for feature in columns for agg in aggregations)
	output = {
		"Name": summary.index.get_level_values("Name").to_numpy(),
		"Frame": summary.index.get_level_values("Frame").to_numpy()
	}
	
	for feature in columns:
		for agg in aggregations:
			output[feature_name(feature, agg)] = summary[(agg, feature)].to_numpy()

	return output, names

def normalize_per_experiment(X_full: np.ndarray, causal: bool = True, min_frames: int = 5) -> np.ndarray:
	"""
	z-score of every feature within each experiment: (x - median) / (1.4826 * MAD).
	Removes gel-to-gel offsets in baseline strain so "unusual for this gel" is what the model sees.

	Arguments:
		X_full: [experiment, frame, feature], NaN padded
		causal (bool): True uses only frames up to t (no future info leaks into frame), False uses the whole experiment
		min_frames (int): frames needed before the expanding estimate is used

	Returns:
		X_norm with the same shape and NaN padding
	"""

	X_norm = np.full_like(X_full, np.nan)
	valid = ~np.isnan(X_full).any(axis=2)

	for e in range(X_full.shape[0]):
		df = pd.DataFrame(X_full[e, valid[e]])
		if causal:
			median = df.expanding(min_frames).median().bfill()
			mad = (df - median).abs().expanding(min_frames).median().bfill()
		else:
			median = df.median()
			mad = (df - median).abs().median()
		scale = (1.4826 * mad).where(1.4826 * mad > 1e-12, 1.0)
		X_norm[e, valid[e]] = ((df - median) / scale).to_numpy()

	return X_norm

def grid_fields(
	data: dict,
	fields: tuple | list,
	grid_size: int = 12
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Bins every frame onto one common grid_size x grid_size grid (pixel coords shared for all experiments).

	Arguments:
		data (dict): raw data from load_data (one row per grid point per frame)
		fields (tuple | list): raw feature keys or keys of DERIVED to bin
		grid_size (int): number of cells per side

	Returns:
		binned [experiment, frame, field, grid_size**2]: cell averages, NaN where a cell has no points
		experiments: experiment names, in the order of the first axis
	"""

	experiments, exp_idx = np.unique(np.asarray(data["Name"]).astype(str), return_inverse=True)
	frames = np.asarray(data["Frame"]).astype(int)
	frame_idx = pd.Series(frames).groupby(exp_idx).rank(method="dense").to_numpy().astype(int) - 1 # 0-based per experiment

	cell = []
	for coord in ("X", "Y"):
		c = np.asarray(data[coord], dtype=float)
		cell.append(np.minimum(((c - c.min()) / (c.max() - c.min()) * grid_size).astype(int), grid_size-1))
	cell = cell[0] * grid_size + cell[1]

	num_experiments, num_frames, num_cells = len(experiments), frame_idx.max() + 1, grid_size**2
	flat = (exp_idx * num_frames + frame_idx) * num_cells + cell
	size = num_experiments * num_frames * num_cells
	counts = np.bincount(flat, minlength=size)

	binned = np.full((num_experiments, num_frames, len(fields), num_cells), np.nan)
	for f, field in enumerate(fields):
		values = DERIVED[field](data) if field in DERIVED else np.asarray(data[field], dtype=float)
		sums = np.bincount(flat, weights=values, minlength=size)
		with np.errstate(invalid="ignore", divide="ignore"):
			binned[:, :, f, :] = np.where(counts > 0, sums / counts, np.nan).reshape(num_experiments, num_frames, num_cells)
	
	return binned, experiments

def fit_pca(binned: np.ndarray, fit_experiments: np.ndarray, n_components: int = 3) -> dict:
	"""
	One PCA per field, on the frames of fit_experiments only (so held-out experiments never shape the components).
	Every experiment gel gets equal total weight, regardless of its number of frames.  Empty cells are filled with training mean,
	so they add nothing.

	Returns:
		pca (dict): "mean" [field, cell], "components" [field, n_components, cell], "explained" [field, n_components]
	"""

	_, _, num_fields, num_cells = binned.shape
	valid = ~np.all(np.isnan(binned), axis=(2, 3)) # [experiment, frame], frame exists
	rows = [(e, t) for e in fit_experiments for t in np.flatnonzero(valid[e])]
	weights = np.array([1 / valid[e].sum() for e, _ in rows])

	mean = np.zeros((num_fields, num_cells))
	components = np.zeros((num_fields, n_components, num_cells))
	explained = np.zeros((num_fields, n_components))
	for f in range(num_fields):
		A = np.array([binned[e, t, f] for e, t in rows]) # [frames, cells]
		observed = ~np.isnan(A)
		counts = weights @ observed
		mean[f] = np.where(counts > 0, (weights @ np.nan_to_num(A)) / np.maximum(counts, 1e-300), 0.0)

		centered = np.where(observed, A - mean[f], 0.0) * np.sqrt(weights)[:, None]
		_, s, Vt = np.linalg.svd(centered, full_matrices=False)
		Vt = Vt[:n_components]
		Vt = Vt * np.where(Vt.sum(axis=1, keepdims=True) < 0, -1.0, 1.0) # sign is arbitrary: fix positive net loading
		components[f], explained[f] = Vt, (s**2 /  np.sum(s**2))[:n_components]
	
	return {"mean": mean, "components": components, "explained": explained}

def project_pca(binned: np.ndarray, pca: dict) -> np.ndarray:
	"""
	Principal component scores of every frame: [experiment, frame, field * n_components], ordered field-major
		(field0_pc1, field0_pc2, ..., field1_pca1, ...), NaN for frames that do not exist (padding).
	"""

	centered = np.nan_to_num(binned - pca["mean"]) # empty cells -> 0 = training mean
	scores = np.einsum("etfc,fkc->etfk", centered, pca["components"]) # e=experiment, t=frame, f=field, c=cell
	valid = ~np.all(np.isnan(binned), axis=(2, 3))
	scores[~valid] = np.nan
	return scores.reshape(*scores.shape[:2], -1)
