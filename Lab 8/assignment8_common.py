"""Shared, from-scratch implementations for AI5707 Assignment 8.

Model algorithms are implemented with Python and NumPy.  Scikit-learn is used
only for evaluation metrics (accuracy and confusion matrix), never for models.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

ROOT = Path.cwd()
if not (ROOT / "DATASETS").exists():
    ROOT = Path(__file__).resolve().parent

DATA = ROOT / "DATASETS"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Reproducible random-number generation (no np.random is used)
# -----------------------------------------------------------------------------

seed = 42


def myRandom():
    """Return one deterministic pseudo-random value in [0, 1) using an LCG."""
    global seed
    a, c, m = 1664525, 1013904223, 2**32
    seed = (a * seed + c) % m
    return seed / m


def myUniform(low, high):
    """Generate a uniform pseudo-random value in [low, high)."""
    return low + (high - low) * myRandom()


def gaussian(mean, sigma, n):
    """Generate n Gaussian values with the polar Box-Muller transform."""
    samples = []

    while len(samples) < n:
        # Draw two values in the unit square and map them to the unit circle.
        u1 = myUniform(-1, 1)
        u2 = myUniform(-1, 1)
        radius_sq = u1**2 + u2**2

        # Points outside the unit circle (and the origin) are rejected.
        if 0 < radius_sq < 1:
            scale = math.sqrt(-2 * math.log(radius_sq) / radius_sq)
            samples.append(mean + sigma * u1 * scale)

            if len(samples) < n:
                samples.append(mean + sigma * u2 * scale)

    return np.asarray(samples)


def deterministic_permutation(n):
    """Return a reproducible shuffled index array using Fisher-Yates."""
    indices = list(range(n))

    for i in range(n - 1, 0, -1):
        j = int(myRandom() * (i + 1))
        indices[i], indices[j] = indices[j], indices[i]

    return np.asarray(indices, dtype=int)


def deterministic_split(X, y, test_fraction=0.2):
    """Split features and labels into train and test sets deterministically."""
    order = deterministic_permutation(len(y))
    split_at = int(len(order) * (1 - test_fraction))
    train_idx, test_idx = order[:split_at], order[split_at:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# -----------------------------------------------------------------------------
# Feature scaling
# -----------------------------------------------------------------------------


def standardize_fit(X):
    """Calculate training-set mean and standard deviation for each feature."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero for constant-valued features.
    std[std == 0] = 1
    return mean, std


def standardize_transform(X, mean, std):
    """Apply previously calculated scaling parameters to a feature matrix."""
    return (np.asarray(X, dtype=float) - mean) / std


# -----------------------------------------------------------------------------
# Principal Component Analysis (PCA) - implemented from scratch
# -----------------------------------------------------------------------------


class PCAFromScratch:
    """PCA using covariance-matrix eigenvalue decomposition."""

    def __init__(self, n_components):
        self.n_components = int(n_components)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_

        # Covariance matrix describes relationships between input features.
        covariance = centered.T @ centered / max(len(X) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Sort eigenvectors by descending eigenvalue (largest variance first).
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        self.components_ = eigenvectors[:, order[: self.n_components]]

        total_variance = max(eigenvalues.sum(), 1e-15)
        self.explained_variance_ratio_ = (
            eigenvalues[: self.n_components] / total_variance
        )
        return self

    def transform(self, X):
        """Project data onto the principal-component directions."""
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# -----------------------------------------------------------------------------
# Linear Discriminant Analysis (LDA) - implemented from scratch
# -----------------------------------------------------------------------------


class LDAFromScratch:
    """Gaussian LDA classifier with a shared within-class covariance matrix."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        sample_count, feature_count = X.shape

        class_means = []
        class_priors = []
        within_scatter = np.zeros((feature_count, feature_count))

        # Estimate each class mean and accumulate within-class scatter.
        for class_label in self.classes_:
            class_samples = X[y == class_label]
            class_mean = class_samples.mean(axis=0)
            class_means.append(class_mean)
            class_priors.append(len(class_samples) / sample_count)

            centered = class_samples - class_mean
            within_scatter += centered.T @ centered

        self.means_ = np.asarray(class_means)
        self.priors_ = np.asarray(class_priors)
        within_scatter /= max(sample_count - len(self.classes_), 1)

        # Pseudo-inverse plus a small diagonal term improves numerical stability.
        regularized = within_scatter + 1e-6 * np.eye(feature_count)
        inverse_covariance = np.linalg.pinv(regularized)

        self.coef_ = self.means_ @ inverse_covariance
        quadratic = np.einsum(
            "ij,jk,ik->i", self.means_, inverse_covariance, self.means_
        )
        self.intercept_ = -0.5 * quadratic + np.log(self.priors_)
        return self

    def predict(self, X):
        scores = np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_
        return self.classes_[np.argmax(scores, axis=1)]


# -----------------------------------------------------------------------------
# K-Nearest Neighbours (KNN) - implemented from scratch
# -----------------------------------------------------------------------------


class KNNFromScratch:
    """KNN classifier using squared Euclidean distance and majority voting."""

    def __init__(self, k=5):
        self.k = int(k)

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        predictions = []

        for sample in np.asarray(X, dtype=float):
            # Compute distance from this sample to every training observation.
            squared_distances = np.sum((self.X_ - sample) ** 2, axis=1)
            nearest_indices = np.argsort(squared_distances)[: self.k]

            # Select the most frequent label among the k nearest neighbours.
            labels, counts = np.unique(
                self.y_[nearest_indices], return_counts=True
            )
            predictions.append(labels[np.argmax(counts)])

        return np.asarray(predictions)


# -----------------------------------------------------------------------------
# Gaussian Mixture Model (GMM) - Expectation-Maximization from scratch
# -----------------------------------------------------------------------------


class GMMFromScratch:
    """Full-covariance Gaussian mixture trained with the EM algorithm."""

    def __init__(self, n_components=1, max_iter=80, tol=1e-4, reg=1e-4):
        self.K = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg = float(reg)

    @staticmethod
    def log_sum_exp(values):
        """Stable log(sum(exp(values))) calculation along each row."""
        row_max = values.max(axis=1, keepdims=True)
        return (
            row_max + np.log(np.exp(values - row_max).sum(axis=1, keepdims=True))
        ).ravel()

    def _log_gaussian_probabilities(self, X):
        """Return log p(x | component k) for every sample and component."""
        sample_count, feature_count = X.shape
        log_probabilities = np.empty((sample_count, self.K))

        for component in range(self.K):
            covariance = self.covariances_[component]
            covariance = covariance + self.reg * np.eye(feature_count)
            sign, log_determinant = np.linalg.slogdet(covariance)
            inverse = np.linalg.pinv(covariance)
            difference = X - self.means_[component]

            # Mahalanobis distance is the quadratic term in the Gaussian density.
            mahalanobis = np.einsum(
                "ij,jk,ik->i", difference, inverse, difference
            )
            log_probabilities[:, component] = -0.5 * (
                feature_count * np.log(2 * np.pi)
                + log_determinant
                + mahalanobis
            )

        return log_probabilities

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        sample_count, feature_count = X.shape

        # Deterministically choose initial component means from the observations.
        initial_indices = deterministic_permutation(sample_count)[: self.K]
        self.means_ = X[initial_indices].copy()

        # Start with the global covariance for each component and equal weights.
        global_covariance = np.atleast_2d(np.cov(X, rowvar=False))
        initial_covariance = global_covariance + self.reg * np.eye(feature_count)
        self.covariances_ = np.repeat(
            initial_covariance[None, :, :], self.K, axis=0
        )
        self.weights_ = np.ones(self.K) / self.K
        previous_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step: calculate the responsibility of each component per sample.
            joint_log_prob = self._log_gaussian_probabilities(X)
            joint_log_prob += np.log(self.weights_ + 1e-15)
            normalizer = self.log_sum_exp(joint_log_prob)
            responsibilities = np.exp(joint_log_prob - normalizer[:, None])
            effective_counts = responsibilities.sum(axis=0) + 1e-10

            # M-step: update mixture weights, means, and covariance matrices.
            self.weights_ = effective_counts / sample_count
            self.means_ = (responsibilities.T @ X) / effective_counts[:, None]

            for component in range(self.K):
                difference = X - self.means_[component]
                weighted = difference.T * responsibilities[:, component]
                self.covariances_[component] = (
                    weighted @ difference / effective_counts[component]
                    + self.reg * np.eye(feature_count)
                )

            # Stop when the average log-likelihood improvement is sufficiently small.
            updated_log_prob = self._log_gaussian_probabilities(X)
            updated_log_prob += np.log(self.weights_ + 1e-15)
            log_likelihood = float(np.mean(self.log_sum_exp(updated_log_prob)))

            if abs(log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = log_likelihood

        self.n_iter_ = iteration + 1
        self.lower_bound_ = log_likelihood
        return self

    def score_samples(self, X):
        """Return the log-likelihood of each sample under the fitted mixture."""
        X = np.asarray(X, dtype=float)
        joint_log_prob = self._log_gaussian_probabilities(X)
        joint_log_prob += np.log(self.weights_ + 1e-15)
        return self.log_sum_exp(joint_log_prob)


def gmm_predict(X_train, y_train, X_test, K=1):
    """Classify samples by fitting one GMM per class and comparing log scores."""
    classes = np.unique(y_train)
    class_models = {}

    for class_label in classes:
        class_data = X_train[y_train == class_label]
        component_count = min(K, len(class_data))
        class_models[class_label] = GMMFromScratch(
            n_components=component_count
        ).fit(class_data)

    class_scores = np.column_stack(
        [class_models[label].score_samples(X_test) for label in classes]
    )
    return classes[np.argmax(class_scores, axis=1)]


# -----------------------------------------------------------------------------
# Support Vector Machine (SVM) - RBF kernel and simplified SMO from scratch
# -----------------------------------------------------------------------------


class SVMBinarySMO:
    """Binary RBF soft-margin SVM trained using simplified SMO updates."""

    def __init__(self, C=1.0, gamma=None, max_passes=5, max_iter=500):
        self.C = float(C)
        self.gamma = gamma
        self.max_passes = int(max_passes)
        self.max_iter = int(max_iter)

    def _kernel(self, A, B):
        """Compute the RBF kernel matrix between rows of A and B."""
        distances = (
            np.sum(A * A, axis=1)[:, None]
            + np.sum(B * B, axis=1)[None, :]
            - 2 * A @ B.T
        )
        squared_distances = np.maximum(distances, 0)
        return np.exp(-self.gamma * squared_distances)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        # SMO expects binary labels encoded as -1 and +1.
        y = np.where(np.asarray(y) == 1, 1.0, -1.0)
        sample_count, feature_count = X.shape

        if self.gamma is None:
            self.gamma = 1 / max(feature_count, 1)

        kernel_matrix = self._kernel(X, X)
        alpha = np.zeros(sample_count)
        bias = 0.0
        passes_without_change = 0
        iteration = 0

        while (
            passes_without_change < self.max_passes
            and iteration < self.max_iter
        ):
            changed_alphas = 0

            for i in range(sample_count):
                error_i = np.dot(alpha * y, kernel_matrix[:, i]) + bias - y[i]

                violates_kkt = (
                    (y[i] * error_i < -1e-3 and alpha[i] < self.C - 1e-6)
                    or (y[i] * error_i > 1e-3 and alpha[i] > 1e-6)
                )
                if not violates_kkt:
                    continue

                # Choose a second observation using the reproducible generator.
                j = (i + 1 + int(myRandom() * (sample_count - 1))) % sample_count
                error_j = np.dot(alpha * y, kernel_matrix[:, j]) + bias - y[j]
                old_alpha_i, old_alpha_j = alpha[i], alpha[j]

                # Bounds preserve the equality constraint after updating alpha_j.
                if y[i] != y[j]:
                    lower = max(0, old_alpha_j - old_alpha_i)
                    upper = min(self.C, self.C + old_alpha_j - old_alpha_i)
                else:
                    lower = max(0, old_alpha_i + old_alpha_j - self.C)
                    upper = min(self.C, old_alpha_i + old_alpha_j)

                if upper - lower < 1e-12:
                    continue

                eta = (
                    2 * kernel_matrix[i, j]
                    - kernel_matrix[i, i]
                    - kernel_matrix[j, j]
                )
                if eta >= 0:
                    continue

                alpha[j] = np.clip(
                    old_alpha_j - y[j] * (error_i - error_j) / eta,
                    lower,
                    upper,
                )

                if abs(alpha[j] - old_alpha_j) < 1e-5:
                    alpha[j] = old_alpha_j
                    continue

                # Update alpha_i so that sum(alpha_i * y_i) remains zero.
                alpha[i] = old_alpha_i + y[i] * y[j] * (old_alpha_j - alpha[j])

                bias_1 = (
                    bias
                    - error_i
                    - y[i] * (alpha[i] - old_alpha_i) * kernel_matrix[i, i]
                    - y[j] * (alpha[j] - old_alpha_j) * kernel_matrix[i, j]
                )
                bias_2 = (
                    bias
                    - error_j
                    - y[i] * (alpha[i] - old_alpha_i) * kernel_matrix[i, j]
                    - y[j] * (alpha[j] - old_alpha_j) * kernel_matrix[j, j]
                )

                if 0 < alpha[i] < self.C:
                    bias = bias_1
                elif 0 < alpha[j] < self.C:
                    bias = bias_2
                else:
                    bias = (bias_1 + bias_2) / 2

                changed_alphas += 1

            # Several unchanged passes indicate that the optimization has settled.
            if changed_alphas == 0:
                passes_without_change += 1
            else:
                passes_without_change = 0
            iteration += 1

        # Only support vectors (non-zero alpha values) are needed for prediction.
        support_mask = alpha > 1e-6
        self.support_vectors_ = X[support_mask]
        self.support_coefficients_ = alpha[support_mask] * y[support_mask]
        self.bias_ = bias
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return (
            self._kernel(X, self.support_vectors_)
            @ self.support_coefficients_
            + self.bias_
        )


class SVMFromScratch:
    """Multiclass SVM using one-vs-rest binary SMO classifiers."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = []

        # Train one binary classifier per class: this class vs all other classes.
        for class_label in self.classes_:
            binary_labels = (y == class_label).astype(int)
            model = SVMBinarySMO(**self.kwargs).fit(X, binary_labels)
            self.models_.append(model)

        return self

    def predict(self, X):
        # The class with the largest decision score is selected for each sample.
        scores = np.column_stack(
            [model.decision_function(X) for model in self.models_]
        )
        return self.classes_[np.argmax(scores, axis=1)]


# -----------------------------------------------------------------------------
# Model comparison helper
# -----------------------------------------------------------------------------


def evaluate_classifiers(
    X_train, y_train, X_test, y_test, pca_k=None, svm_limit=1200
):
    """Train all four classifiers and return their test accuracies.

    Standardization and optional PCA are fitted using training data only to
    avoid leaking information from the test set into model training.
    """
    mean, std = standardize_fit(X_train)
    train_scaled = standardize_transform(X_train, mean, std)
    test_scaled = standardize_transform(X_test, mean, std)

    if pca_k is not None:
        component_count = min(pca_k, train_scaled.shape[1], len(train_scaled) - 1)
        pca = PCAFromScratch(component_count).fit(train_scaled)
        train_scaled = pca.transform(train_scaled)
        test_scaled = pca.transform(test_scaled)

    # The kernel SVM stores an O(n^2) matrix, so use a deterministic subset
    # for SVM training on large datasets. Other models use all training rows.
    if len(train_scaled) > svm_limit:
        subset = deterministic_permutation(len(train_scaled))[:svm_limit]
        svm_train_X = train_scaled[subset]
        svm_train_y = np.asarray(y_train)[subset]
    else:
        svm_train_X = train_scaled
        svm_train_y = np.asarray(y_train)

    predictions = {
        "GMM": gmm_predict(train_scaled, y_train, test_scaled),
        "LDA": LDAFromScratch().fit(train_scaled, y_train).predict(test_scaled),
        "KNN": KNNFromScratch(k=5).fit(train_scaled, y_train).predict(test_scaled),
        "SVM": SVMFromScratch(C=2.0, max_passes=4, max_iter=300)
        .fit(svm_train_X, svm_train_y)
        .predict(test_scaled),
    }

    # sklearn.metrics is used here only to calculate evaluation scores.
    return {
        model_name: accuracy_score(y_test, predicted_labels)
        for model_name, predicted_labels in predictions.items()
    }


def dtw_distance(A, B):
    """Compute Dynamic Time Warping distance between two feature sequences."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # D[i, j] stores the minimum cumulative alignment cost up to (i, j).
    cost_matrix = np.full((len(A) + 1, len(B) + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            local_cost = np.linalg.norm(A[i - 1] - B[j - 1])
            previous_cost = min(
                cost_matrix[i - 1, j],
                cost_matrix[i, j - 1],
                cost_matrix[i - 1, j - 1],
            )
            cost_matrix[i, j] = local_cost + previous_cost

    # Normalize by the combined sequence length for easier comparison.
    return cost_matrix[-1, -1] / (len(A) + len(B))
