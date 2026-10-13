# gel-tsml

gel-tsml (time-series machine learning) implements the following machine learning models for active gel contraction experiments:
- Predicting when a contraction will occur: linear regression including regularization (Lasso, Ridge, or ElasticNet).
- Predicting what type of contraction will occur: logistic regression.

## Quickstart

To create the conda environment, simply ensure that Miniconda is installed and call the `create_conda_env` executable:
```Bash
cd gel-tsml
./create_conda_env
conda activate gel-tsml
```

### Required packages
- `python`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pip`
- `jax[cuda12]` / `jax`

## Files
- `gel_io.py`: converts raw .txt/.npz data into a dict for analysis.
- `gel_ts.py`: sets up time-series data.  Converts prior dict into `X_full[experiment, frame, feature]`.
- `spatial_features.py`: performs spatial reductions, if applicable, such as averaging, determining percentiles, and PCA (soon).
- `gel_windows.py`: collects data into a sliding time window, classifying contraction "events" and target values.
- `jax_linear.py`/`sklearn_linear.py`: equivalent regression models with `Ridge`, `Lasso`, or `ElasticNet` regressions for event predictions, and `LogisticRegression` for event classifications.
- `gel_cv.py`: performs the cross-validation splits for given experimental data.
- `run_regression.py`: declare settings for a given calculation, perform modeling, and generate data/plots.

Uses scikit.learn and JAX to determine a linear regression model (with regularizations) to predict active gel contractions.
