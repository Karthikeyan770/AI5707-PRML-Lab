# AI5707 Assignment 4 — Multivariate Gaussian Density

## Contents
- `assignment4.py` — complete reproducible implementation.
- `report.docx` — submission-style report with derivations, numerical results, observations, and figures.
- `figures/` — generated plots.
- `data/` — generated CSV datasets.
- `numerical_results.npz` — NumPy numerical results.

## Important parameter note
The Assignment 4 sheet says to use `mu1` and `mu2` "as in the first assignment", but the Assignment 4 PDF itself does not specify their numerical values. The default implementation therefore uses:
- `mu1 = -2`
- `mu2 = 2`
- `sigma^2 = 1` for the isotropic base data
- random seed = 42
- N = 2000 samples

If your Assignment 1 used different means, change `MU1` and `MU2` near the top of `assignment4.py` and rerun.

## Run
```bash
pip install numpy matplotlib python-docx
python assignment4.py
```

## Main mathematical ideas
For a 2-D Gaussian:
`p(x) = 1 / ((2*pi)*sqrt(|Sigma|)) * exp(-0.5*(x-mu)^T Sigma^(-1) (x-mu))`

Constant-density contours are therefore constant values of:
`(x-mu)^T Sigma^(-1) (x-mu) = c`

For eigendecomposition:
`Sigma = Q Lambda Q^T`

The eigenvectors give the principal directions and the eigenvalues determine squared axis lengths. For a contour level `c`, the semi-axis lengths are:
`sqrt(c * lambda_i)`.

For classification with equal priors:
`g(x) = log p(x|w1) - log p(x|w2)`.
Equal covariance matrices cancel the quadratic term and give a linear boundary; unequal covariance matrices generally produce a quadratic boundary.

## Viva note
Be able to explain why:
1. isotropic covariance -> circular contours;
2. diagonal unequal covariance -> axis-aligned ellipses;
3. nonzero covariance -> rotated ellipses;
4. eigenvectors are directions of the principal axes;
5. eigenvalues control the squared spread along those axes;
6. QDA arises when class covariance matrices differ.
