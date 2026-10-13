import numpy as np
import pandas as pd
from pathlib import Path

def import_txt(txt_path: Path | str) -> dict:
	"""
	Import .txt data into a dictionary.

	Arguments:
		txt_path (Path | str): Path to the .txt file.

	Returns:
		data (dict): Dictionary containing the data.  Keys are the original csv column headers.
	"""

	# Specify non-float dtypes, the float dtypes are determined automatically
	dtypes = {"X": int, "Y": int, "Name": str, "Frame": str, "Region": str}

	df = pd.read_csv(txt_path, sep="\t", dtype=dtypes)

	# Remove whitespace, leading "\", and trailing "_frame_" from keys
	df.columns = df.columns.str.strip().str.lstrip("\\").str.removesuffix("_frame_")

	return {col: df[col].to_numpy() for col in df.columns}

def txt_to_npz(txt_path: Path | str, npz_path: Path | str | None = None):
	"""
	Compresses raw .txt data into .npz

	Arguments:
		txt_path (Path or str): Path to .txt file
		npz_path (Path or str or None, optional): Path where to save .npz file.  Defaults to text_path with .npz extension
	"""

	if npz_path is None:
		npz_path = Path(txt_path).with_suffix(".npz")

	data = import_txt(txt_path)
	np.savez_compressed(npz_path, **data) # **data sets column headers from csv as data keys

def load_data(file_path: Path | str):
	"""
	Imports .txt or .npz data into a dictionary.

	Arguments:
		file_path (Path or str): Path to the .txt or .npz file
	
	Returns:
		data (dict): Dictionary containing the data.  Keys are the original csv column headers.
	"""

	file_path = Path(file_path)

	if not Path(file_path).is_file():
		raise ValueError(f"load_data: declared {file_path=} does not exist...")

	if file_path.suffix == ".txt":
		return import_txt(file_path)
	elif file_path.suffix == ".npz":
		with np.load(file_path, allow_pickle=True) as data:
			return {key: data[key] for key in data.files}
	else:
		raise ValueError(f"load_data: file_path must be .txt or .npz, received: {file_path.suffix}")
