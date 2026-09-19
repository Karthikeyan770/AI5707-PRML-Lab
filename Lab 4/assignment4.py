"""
AI5707 Assignment 4 — Multivariate Gaussian Density
Author: Student submission template
Default configuration is reproducible (seed=42).

Run:
    python assignment4.py

The script:
1. Generates two 1-D Gaussian datasets for the isotropic case.
2. Forms 2-D samples and estimates covariance/eigenpairs.
3. Repeats for diagonal covariance.
4. Repeats for a full covariance matrix.
5. Generates two Gaussian classes and plots decision boundaries.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
N = 2000

# Assignment 4 refers to means from Assignment 1 but does not state their
# numerical values. Change these two values if your Assignment 1 used others.
MU1, MU2 = -2.0, 2.0
MU = np.array([MU1, MU2], dtype=float)

C_ISO = np.eye(2)
C_DIAG = np.array([[4.0, 0.0], [0.0, 1.0]])
C_FULL = np.array([[4.0, 1.2], [1.2, 1.5]])

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
DATADIR = ROOT / "data"
FIGDIR.mkdir(exist_ok=True)
DATADIR.mkdir(exist_ok=True)

def box_muller_polar(n, rng):
    out = []
    while len(out) < n:
        u1, u2 = rng.uniform(-1.0, 1.0, size=2)
        s = u1*u1 + u2*u2
        if s == 0.0 or s >= 1.0:
            continue
        k = np.sqrt(-2.0*np.log(s)/s)
        out.append(u1*k)
        if len(out) < n:
            out.append(u2*k)
    return np.asarray(out[:n])

def standard_normal_matrix(n, rng):
    return np.column_stack([box_muller_polar(n, rng),
                             box_muller_polar(n, rng)])

def generate_gaussian(mu, cov, n, rng):
    
    """Y = mu + L Z, with LL^T = cov."""
    z = standard_normal_matrix(n, rng)
    L = np.linalg.cholesky(cov)
    return mu + z @ L.T

def sample_covariance(X):
    return np.cov(X, rowvar=False, ddof=1)

def eig_sorted(C):
    values, vectors = np.linalg.eigh(C)
    idx = np.argsort(values)[::-1]
    return values[idx], vectors[:, idx]

def save_csv(path, X):
    np.savetxt(path, X, delimiter=",", header="x1,x2", comments="")

def mahalanobis_squared(X, mu, cov):
    d = X - mu
    return np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d)

def gaussian_logpdf(X, mu, cov):
    d = X - mu
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    inv = np.linalg.inv(cov)
    q = np.einsum("...i,ij,...j->...", d, inv, d)
    return -0.5 * (2*np.log(2*np.pi) + logdet + q)

def plot_constant_curves(X, mu, cov, title, filename):
    values, vectors = eig_sorted(cov)
    theta = np.linspace(0, 2*np.pi, 500)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:,0], X[:,1], s=7, alpha=0.20, label="samples")
    ax.scatter([mu[0]], [mu[1]], marker="x", s=90, linewidths=2,
               label="estimated mean")
    for c, style in [(1.0, "-"), (2.0, "--"), (4.0, ":")]:
        circle = np.vstack([np.cos(theta), np.sin(theta)]) * np.sqrt(c)
        pts = mu[:, None] + vectors @ (np.sqrt(values)[:, None] * circle)
        ax.plot(pts[0], pts[1], linestyle=style, label=fr"$c={c:g}$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / filename, dpi=180)
    plt.close(fig)

def plot_eigenvectors(X, mu, cov, title, filename):
    values, vectors = eig_sorted(cov)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:,0], X[:,1], s=7, alpha=0.20)
    ax.scatter([mu[0]], [mu[1]], marker="x", s=90, linewidths=2)
    for i in range(2):
        v = vectors[:, i] * np.sqrt(values[i])
        for sign in (1, -1):
            ax.arrow(mu[0], mu[1], sign*v[0], sign*v[1],
                     width=0.01, length_includes_head=True,
                     head_width=0.12, head_length=0.16)
        ax.text(mu[0]+v[0], mu[1]+v[1],
                fr"$\lambda_{i+1}={values[i]:.3f}$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / filename, dpi=180)
    plt.close(fig)

def decision_grid(mu1, c1, mu2, c2, lim=(-7, 7), ngrid=500):
    x = np.linspace(lim[0], lim[1], ngrid)
    y = np.linspace(lim[0], lim[1], ngrid)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)
    g = gaussian_logpdf(grid, mu1, c1) - gaussian_logpdf(grid, mu2, c2)
    return xx, yy, g

def plot_boundary(X1, X2, mu1, c1, mu2, c2, title, filename):
    xx, yy, g = decision_grid(mu1, c1, mu2, c2)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X1[:,0], X1[:,1], s=7, alpha=0.18, label=r"$\omega_1$")
    ax.scatter(X2[:,0], X2[:,1], s=7, alpha=0.18, label=r"$\omega_2$")
    ax.contour(xx, yy, g, levels=[0], linewidths=2)
    ax.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]],
               marker="x", s=90, linewidths=2)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / filename, dpi=180)
    plt.close(fig)

def main():
    rng = np.random.default_rng(SEED)

    # Q1: isotropic covariance
    D1 = MU1 + box_muller_polar(N, rng)
    D2 = MU2 + box_muller_polar(N, rng)
    X = np.column_stack([D1, D2])
    mu_hat = X.mean(axis=0)
    cov_hat = sample_covariance(X)
    values, vectors = eig_sorted(cov_hat)
    save_csv(DATADIR / "q1_isotropic_data.csv", X)
    plot_constant_curves(X, mu_hat, cov_hat,
                         "Q1 — Isotropic covariance",
                         "q1_constant_density_curves.png")
    plot_eigenvectors(X, mu_hat, cov_hat,
                      "Q1 — Eigenvectors and principal axes",
                      "q1_eigenvectors.png")

    print("\nQ1")
    print("Estimated mean:\n", mu_hat)
    print("Estimated covariance:\n", cov_hat)
    print("Eigenvalues:", values)
    print("Eigenvectors (columns):\n", vectors)

    # Q2: diagonal covariance
    rng = np.random.default_rng(SEED)
    X0 = np.column_stack([MU1 + box_muller_polar(N, rng),
                          MU2 + box_muller_polar(N, rng)])
    X = MU + (X0 - MU) @ np.diag(np.sqrt(np.diag(C_DIAG)))
    mu_hat = X.mean(axis=0)
    cov_hat = sample_covariance(X)
    values, vectors = eig_sorted(cov_hat)
    save_csv(DATADIR / "q2_diagonal_data.csv", X)
    plot_constant_curves(X, mu_hat, cov_hat,
                         "Q2 — Diagonal covariance",
                         "q2_diagonal_curves.png")
    print("\nQ2")
    print("Estimated mean:\n", mu_hat)
    print("Estimated covariance:\n", cov_hat)
    print("Eigenvalues:", values)

    # Q3: full covariance
    rng = np.random.default_rng(SEED)
    X = generate_gaussian(MU, C_FULL, N, rng)
    mu_hat = X.mean(axis=0)
    cov_hat = sample_covariance(X)
    values, vectors = eig_sorted(cov_hat)
    save_csv(DATADIR / "q3_full_covariance_data.csv", X)
    plot_constant_curves(X, mu_hat, cov_hat,
                         "Q3 — Full covariance",
                         "q3_full_covariance_curves.png")
    plot_eigenvectors(X, mu_hat, cov_hat,
                      "Q3 — Eigenvectors and principal axes",
                      "q3_eigenvectors.png")
    print("\nQ3")
    print("Estimated mean:\n", mu_hat)
    print("Estimated covariance:\n", cov_hat)
    print("Eigenvalues:", values)

    # Q4: two classes
    mu_a = np.array([-2.0, 0.0])
    mu_b = np.array([2.0, 1.0])
    cov_a = np.array([[1.2, 0.3], [0.3, 0.8]])
    cov_b = np.array([[1.5, -0.4], [-0.4, 1.0]])
    rng = np.random.default_rng(SEED)
    Y1 = generate_gaussian(mu_a, cov_a, N, rng)
    Y2 = generate_gaussian(mu_b, cov_b, N, rng)
    save_csv(DATADIR / "q4_class1.csv", Y1)
    save_csv(DATADIR / "q4_class2.csv", Y2)

    plot_boundary(Y1, Y2, mu_a, cov_a, mu_b, cov_b,
                  "Q4 — Unequal covariance: quadratic boundary",
                  "q4_unequal_covariance_boundary.png")

    shared = np.array([[1.4, 0.25], [0.25, 1.0]])
    plot_boundary(Y1, Y2, mu_a, shared, mu_b, shared,
                  "Q4 — Equal covariance: linear boundary",
                  "q4_equal_covariance_boundary.png")

    diag_a = np.diag([1.0, 0.6])
    diag_b = np.diag([2.0, 1.2])
    plot_boundary(Y1, Y2, mu_a, diag_a, mu_b, diag_b,
                  "Q4 — Diagonal unequal covariance: quadratic boundary",
                  "q4_diagonal_unequal_boundary.png")

if __name__ == "__main__":
    main()
