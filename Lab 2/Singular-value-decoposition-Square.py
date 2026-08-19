import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert it to grayscale
img = cv2.imread("cat_15.jpeg")
A = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

# Perform SVD
U, S, Vh = np.linalg.svd(A, full_matrices=False)

# Vh is the Hermitian transpose (V^H) returned by NumPy

# k values used for the experiment
ks = [20, 90, 100]

print("SVD k values:", ks)

# Reconstruct the image for each k
for k in ks:

    # Rank-k approximation: A_k = U_k Sigma_k V_k^H
    Ak = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]

    # Calculate reconstruction error
    error_img = np.abs(A - Ak)
    error = np.linalg.norm(A - Ak, "fro")

    print(f"k = {k}, Frobenius error = {error:.4f}")

    # Display original, reconstructed, and error images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(Ak, 0, 255), cmap="gray")
    plt.title(f"SVD, k = {k}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(error_img, cmap="gray")
    plt.title("Error")
    plt.axis("off")

    plt.tight_layout()
    plt.show()