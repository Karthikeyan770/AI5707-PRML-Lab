# AI5707 Assignment 8 — Team 16

## Notebooks
1. `01_Synthetic.ipynb` — GMM, LDA, KNN, SVM and PCA.
2. `02_MNIST.ipynb` — same classifiers and PCA for the supplied MNIST-labelled pickle.
3. `03_Image_Dataset.ipynb` — classifiers and PCA on image features.
4. `04_TI_Digits_DTW.ipynb` — sequential digit recognition using DTW.
5. `05_Combined_Comparison.ipynb` — combines generated result CSVs.

## Model implementation and evaluation
`assignment8_common.py` implements GMM with EM (three components per class by default), LDA, KNN, binary RBF-SVM with simplified SMO and one-vs-rest multiclass classification, PCA using covariance eigen-decomposition, the custom RNG, and DTW. GMM predictions include class-prior probabilities, matching the reference test implementation. **No scikit-learn model, estimator, or preprocessing implementation is used.** Python and NumPy are used to implement the algorithms. `sklearn.metrics` is used only for evaluation metrics such as accuracy and confusion matrix. No `np.random` calls are used.

## Randomness
Seed=42; LCG constants a=1664525, c=1013904223, m=2**32; provided polar Box-Muller Gaussian method. Deterministic Fisher-Yates permutation uses myRandom().

## Run
`pip install -r requirements.txt`, open each notebook from this project folder and run top-to-bottom. Run notebooks 01–04 before `05_Combined_Comparison.ipynb`. Results are written to `outputs/`.

## Dataset note
The supplied `team16_mnist/data.pkl` appears identical to `team16_Synthetic/data.pkl` in both X and y arrays. It is retained and processed as supplied, but its output cannot be represented as genuine MNIST until the course-provided file is confirmed.

## Runtime note
The handwritten SMO implementation uses an O(n^2) kernel and trains on a deterministic subset (up to 1,200 examples) to keep execution practical. GMM, LDA and KNN use the available training data.
