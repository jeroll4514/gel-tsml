import numpy as np

from pathlib import Path

from gel_io import load_data
from spatial_features import summarize_frames, grid_fields, fit_pca, project_pca

class time_series:
	"""
	Class to input .npz or .txt data, converting it into formatted time-series data.

	Attributes:
		X_full: array of features per frame for each experiment [experiment, frame, feature]
			NaN padded past the end of shorter experiments
		experiments: experiment names, in the order of X_full's first axis
		features: feature names, in the order of X_full's last axis
			("summary" expands these, e.g. "E_max_p99", "pca" appends the component scores, e.g. "E_max_pc1")
		spatial/aggregations/derived: how the spatial dimensions were reduced
		pca_fields/n_components/grid_size: "pca" only, which fields get a PCA and how
		binned: "pca" only, fields binned onto the common grid [experiment, frame, field, cell]
		pca: "pca" only, the most recent fit (see spatial_features.fit_pca)
	"""

	def __init__(
		self,
		file_path: Path | str,
		features: tuple | list,
		spatial: str = "average",
		aggregations: tuple | list = ("mean", "std", "p99"),
		derived: tuple | list = ("disp",),
		pca_fields: tuple | list = ("E_max", "I1", "Henky", "disp"),
		n_components: int = 3,
		grid_size: int = 12
	):
		"""
		Arguments:
			file_path (Path | str): Path to .txt or .npz file.
			features (tuple | list): which features (keys for dict) to save for formatting.
			spatial (str | None): how to handle spatial degrees of freedom. "average", "summary", "PCA", or None
				"average" is "summary" with aggregations=("mean",) and no derived fields
				"pca" is "summary" plus the principal component scores of pca_fields
			aggregations (tuple | list): "summary" only, per-frame statistics (see spatial_features.AGGREGATIONS)
			derived (tuple | list): "summary" only, extra per-point fields (see spatial_features.DERIVED)
			pca_fields (tuple | list): "pca" only, raw features or DERIVED keys to run a PCA on (one PCA each)
			n_components (int): "pca" only, components kept per field
			grid_size (int): "pca" only, the binning grid is grix_size x grid_size cells
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
		self.pca_fields = tuple(pca_fields)
		self.n_components = n_components
		self.grid_size = grid_size

		data = load_data(file_path)
		self.experiments = np.unique(data["Name"].astype(str))

		self.X_full = self._construct_X(data)

		if spatial == "pca":
			self.summary_features = self.features
			self.X_summary = self.X_full
			self.binned, _ = grid_fields(data, self.pca_fields, grid_size)
			self.pca_features = tuple(f"{field}_pc{k+1}" for field in self.pca_fields for k in range(n_components))
			self.features = self.summary_features + self.pca_features
			self.X_full = self.pca_X() # PCA fit on every experiment: fine for exploring, see pca_X for CV

		del data # frees data from memory since X_full is just data reformatted

	def pca_X(self, fit_on: np.ndarray | list | None = None) -> np.ndarray:
		"""
		Summary features plus PCA scores, with the PCA fit only on the experiments in fit_on.

		For cross-validation pass every experiment EXCEPT the test one, otherwise the held-out experiment shapes
		the components it is scored on (unsupervised, but still leakage).

		Arguments:
			fit_on (array | list | None): experiment indices to fit the PCA on, None = all experiments

		Returns:
			X_full [experiment, frame, summary features + pca features], features in self.features order
		"""

		fit_on = np.arange(len(self.experiments)) if fit_on is None else np.asarray(fit_on)
		self.pca = fit_pca(self.binned, fit_on, self.n_components)
		return np.concatenate([self.X_summary, project_pca(self.binned, self.pca)], axis=2)

	def _reduce_spatial(self, data: dict) -> dict:
		""" Collapses the spatial grid to one row per experiment/frame, and updates self.features """

		# "pca" builds its summary features here too; the component scores are added in __init__
		# Replaces self.features with the summary names, e.g. ("E_max", "E_max_std", "E_max_p99", ...)
		output, self.features = summarize_frames(data, self.features, self.aggregations, self.derived)
		return output

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
