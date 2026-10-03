# Lab 5 — Maximum Likelihood Parameter Estimation

## Objective
Estimate parameters using Maximum Likelihood Estimation (MLE) on:
1. the noisy polynomial data generated/used in the previous assignment (`Lab 3/noisy_16.txt`), and
2. a real regression dataset (scikit-learn Diabetes dataset).

## Contents
- `main.ipynb` — derivation, NumPy implementation, experiments, plots, interpretation and viva questions.
- `requirements.txt` — Python dependencies.

## Main result
For Gaussian observation noise,

\[
\mathbf{w}_{ML}=(\Phi^T\Phi)^{-1}\Phi^T\mathbf{t}
\]

and

\[
\sigma^2_{ML}=\frac{1}{N}\sum_{n=1}^{N}(t_n-\hat t_n)^2.
\]

The notebook uses `np.linalg.lstsq` for the numerical implementation because it is more stable than explicitly computing a matrix inverse.

## Running the notebook
Open `main.ipynb` from the `Lab 5` directory. The previous-assignment dataset is loaded using the relative path `../Lab 3/noisy_16.txt`.
