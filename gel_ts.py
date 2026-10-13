import numpy as np

from pathlib import Path

from gel_io import load_data
from spatial_features import summarize_frames

class time_series:
	"""
	Class to input .npz or .txt data, converting it into formatted time-series data.

	Attributes:
		X_full: array of features per frame for each experiment [experiment, frame, feature]
			NaN padded past the end of shorter experiments
		experiments: experiment names, in the order of X_full's first axis
		features: feature names, in the order of X_full's last axis ("summary" expands these, e.g. "E_max_p99")
		spatial/aggregations/derived: how the spatial dimensions were reduced
	"""

	def __init__(
		self,
		file_path: Path | str,
		features: tuple | list,
		spatial: str = "average",
		aggregations: tuple | list = ("mean", "std", "p99"),
		derived: tuple | list = ("disp",),
	):
		"""
		Arguments:
			file_path (Path | str): Path to .txt or .npz file.
			features (tuple | list): which features (keys for dict) to save for formatting.
			spatial (str | None): how to handle spatial degrees of freedom. "average", "summary", "PCA", or None
			aggregations (tuple | list): "summary" only, per-frame statistics (see spatial_features.AGGREGATIONS)
			derived (tuple | list): "summary" only, extra per-point fields (see spatial_features.DERIVED)
		"""

		if not isinstance(features, (tuple, list)):
			raise TypeError(f"features must be a tuple or list, received: {type(features).__name__}")
		spatial = spatial.lower()
		if spatial not in ("average", "summary", "pca"):
			raise ValueError(f"spatial must be 'average', 'summary', 'PCA', or None, received: {spatial}")

		self.features = tuple(features)
		self.spatial = spatial
		self.aggregations = ("mean",) if spatial == "average" else tuple(aggregations)
		self.derived = () if spatial == "average" else tuple(derived)

		data = load_data(file_path)
		self.experiments = np.unique(data["Name"].astype(str))

		self.X_full = self._construct_X(data)

		del data # frees data from memory since X_full is just data reformatted

	def _reduce_spatial(self, data: dict) -> dict:
		""" Collapses the spatial grid to one row per experiment/frame, and updates self.features """

		if self.spatial in ("average", "summary"):
			# Replaces self.features with the summary names, e.g. ("E_max", "E_max_std", "E_max_p99", ...)
			output, self.features = summarize_frames(data, self.features, self.aggregations, self.derived)
			return output

		if self.spatial == "pca":
			raise NotImplementedError("PCA not implemented yet")

	def _construct_X(self, data: dict) -> np.ndarray:
		""" Takes data dict and turns it into ordered time-series data X[experiment, frame, feature] """

		data = self._reduce_spatial(data)

		names = data["Name"].astype(str)
		frames = data["Frame"].astype(int)

		num_experiments = len(self.experiments)
		num_frames = len(np.unique(frames))
		num_features = len(self.features)

		X_full = np.full((num_experiments, num_frames, num_features), np.nan, dtype=float)
		for name_idx, name in enumerate(self.experiments):
			name_mask = name == names # only picks out current name in data
			for frame_idx, frame in enumerate(np.unique(frames[name_mask])):
				mask = name_mask & (frame == frames) # only picks out current frame data
				if np.count_nonzero(mask) != 1:
					raise ValueError(f"Expected one row per experiment/frame, got {name=} and {frame=}.")

				for feature_idx, feature in enumerate(self.features):
					X_full[name_idx, frame_idx, feature_idx] = data[feature][mask].item() # .item() here since len=1

		return X_full
