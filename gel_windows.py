
import numpy as np

from dataclasses import dataclass, field

OUTCOMES = ("none", "tear", "detach")
OUTCOME_KEYWORDS = {"rupture": "tear", "rip": "tear", "slide": "detach", "control": "none"}

def outcome_from_name(name: str) -> str:
	""" Maps an experiment name to an outcome in OUTCOMES """

	for keyword, outcome in OUTCOME_KEYWORDS.items():
		if keyword in name.lower():
			return outcome
	raise ValueError(f"outcome_from_name: no outcome keyword found in {name=}")

def experiment_lengths(X_full: np.ndarray) -> np.ndarray:
	""" Number of valid (non-NaN-padded) frames per experiment, X_full[experiment, frame, feature] """
	valid = ~np.isnan(X_full).any(axis=2)
	return valid.sum(axis=1)

def detect_events(
	X_full: np.ndarray,
	features: tuple | list,
	event_feature: str = "E_max",
	threshold: float = 0.1,
	event_frames: dict | None = None,
	experiments: np.ndarray | None = None
) -> np.ndarray:
	"""
	Marks contraction event onsets.  An onset is a rising edge of event_feature through threshold.

	Arguments:
		X_full: [experiment, frame, feature]
		features: feature names matching the last axis of X_full
		event_feature/threshold: signal and level that define "contracting"
		event_frames (dict | None, optional): {experiment name: [onset frame indices]}.  Overrides the detector for the listed experiments.
		experiments (np.ndarray | None, optional): experiment names (needed with event_frames)

	Returns:
		onsets (np.ndarray bool): [experiment, frame]
	"""

	if event_feature not in features:
		raise ValueError(f"detect_events: {event_feature=} must be one of the loaded features {features}.")

	signal = X_full[:, :, list(features).index(event_feature)]
	above = np.nan_to_num(signal, nan=-np.inf) > threshold
	onsets = np.zeros_like(above)
	onsets[:, 1:] = above[:, 1:] & ~above[:, :-1]

	if event_frames is not None:
		names = list(np.asarray(experiments).astype(str))
		for name, frames in event_frames.items():
			onsets[names.index(name)] = False
			onsets[names.index(name), list(frames)] = True

	return onsets

@dataclass
class window_set:
	"""
	Sliding-window samples built from X_full.  Every array shares the first axis (one row per window).

	Attributes:
		X: flattened look-back window [sample, lookback * feature], ordered (oldest frame --> newest frame)
		exp_idx: experiment index from each sample
		t: frame index of the newest frame in each window
		targets: {
			"forcast": features at t + horizon},
			"event_soon": one-hot [no onset, onset] within frames (t, t + horizon),
			"time_to_event": frames until next onset after t, capped at max_wait
			"outcome": one-hot experiment outcome (consistant within an experiment)
			}
		experiments/features/lookback/horizon: bookkeeping
	"""

	X: np.ndarray
	exp_idx: np.ndarray
	t: np.ndarray
	targets: dict = field(default_factory=dict)
	experiments: np.ndarray | None = None
	features: tuple = ()
	lookback: int = 1
	horizon: int = 1

	@property
	def num_experiments(self):
		return len(self.experiments)

	def feature_names(self) -> list:
		""" Column names of X, e.g. 'Henky[t-2]' """
		names = []
		for lag in range(self.lookback):
			for f in self.features:
				if lag < self.lookback - 1:
					time_label = f"t-{self.lookback - 1 - lag}"
				else:
					time_label = "t"
				names.append(f"{f}[{time_label}]")
		return names

	def persistence(self) -> np.ndarray:
		""" Naive forcast baseline: features at t + horizon = features at t """
		return self.X[:, -len(self.features):]

def make_windows(
	X_full: np.ndarray,
	experiments: np.ndarray,
	features: tuple | list,
	lookback: int,
	horizon: int,
	onsets: np.ndarray | None = None,
	max_wait: int | None = None
) -> window_set:
	"""
	Turns X_full[experiment, frame, feature] into supervised (window -> target) samples.

	Arguments:
		lookback: number of past frames in each window (training_length in frames).
		horizon: how many frames ahead to forecast (prediction_length in frames).
		onsets: output of detect_events, needed for the event targets.
		max_wait: cap for time_to_event (defaults to 3*horizon)
	"""

	max_wait = 3*horizon if max_wait is None else max_wait
	lengths = experiment_lengths(X_full) # list storing the number of frames per experiment
	num_features = X_full.shape[2]

	X, exp_idx, t_last, forecast, soon, wait, outcome = [], [], [], [], [], [], []
	for e, length in enumerate(lengths):
		if length < lookback + horizon:
			print(f"make_windows: WARNING, {experiments[e]} has {length} frames < lookback + horizon, skipped.")
			continue

		label = OUTCOMES.index(outcome_from_name(str(experiments[e])))
		for t in range(lookback - 1, length - horizon):
			X.append(X_full[e, t - lookback + 1 : t + 1].reshape(-1))
			exp_idx.append(e)
			t_last.append(t)
			forecast.append(X_full[e, t + horizon])
			outcome.append(label)

			if onsets is not None:
				soon.append(onsets[e, t + 1 : t + horizon + 1].any())
				future = np.flatnonzero(onsets[e, t + 1 : length]) # onset past the window, within recorded frames
				wait.append(min(future[0] + 1, max_wait) if future.size else max_wait)

	targets = {
		"forecast": np.asarray(forecast).reshape(-1, num_features),
		"outcome": np.eye(len(OUTCOMES))[outcome]
	}
	if onsets is not None:
		targets["event_soon"] = np.eye(2)[np.asarray(soon, dtype=int)]
		targets["time_to_event"] = np.asarray(wait, dtype=float)[:, None]

	return window_set(
		X=np.asarray(X), exp_idx=np.asarray(exp_idx), t=np.asarray(t_last), targets=targets,
		experiments=np.asarray(experiments), features=tuple(features), lookback=lookback,
		horizon=horizon
	)
