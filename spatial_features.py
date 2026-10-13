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
