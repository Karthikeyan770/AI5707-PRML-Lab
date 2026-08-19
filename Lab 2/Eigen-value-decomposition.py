import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
 
# Read the image and make it grayscale 
img = cv2.imread("cat_16.jpeg") 
A = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) 
 
if A.shape[0] != A.shape[1]: 
    raise ValueError("EVD needs a square image.") 
 
n = A.shape[0] 
 
# Find eigenvalues and eigenvectors 
 
eigval, Q = np.linalg.eig(A) 
 
# Put the larger eigenvalues first 
idx = np.argsort(np.abs(eigval))[::-1] 
eigval = eigval[idx] 
Q = Q[:, idx] 
 
 
# Keep complex conjugate eigenvalues together 
 
used = set() 
groups = [] 
 
for i in range(n): 
 
    if i in used: 
        continue 
 
    if abs(eigval[i].imag) < 1e-6: 
        groups.append([i]) 
        used.add(i) 
 
    else: 
        target = np.conj(eigval[i]) 
        pair = None 
 
        for j in range(n): 
            if j != i and j not in used: 
                if abs(eigval[j] - target) < 1e-6: 
                    pair = j 
                    break 
 
        if pair is not None: 
            groups.append([i, pair]) 
            used.update([i, pair]) 
        else: 
            groups.append([i]) 
            used.add(i) 
 
# Arrange the pairs in descending order 
 
groups.sort( 
    key=lambda g: max(np.abs(eigval[g])), 
    reverse=True 
) 
 
order = [i for group in groups for i in group] 
 
eigval = eigval[order] 
Q = Q[:, order] 
 
# These are the k values chosen for the experiment 
 
chosen_k = [10, 50, 100] 
 
# Make sure a k does not split a conjugate pair 
 
valid_k = [] 
count = 0 
 
while count < n: 
 
    if abs(eigval[count].imag) < 1e-6: 
        count += 1 
    else: 
        count += 2 
 
    valid_k.append(count) 
 
ks = [ 
    min(valid_k, key=lambda x: abs(x - k)) 
    for k in chosen_k 
] 
 
print("EVD k values:", ks) 
 
# Inverse of Q 
Q_inv = np.linalg.inv(Q) 
 
 
# Reconstruct the image 
for k in ks: 
 
    Lambda = np.zeros((n, n), dtype=complex) 
    Lambda[:k, :k] = np.diag(eigval[:k]) 
 
    Ak = np.real(Q @ Lambda @ Q_inv) 
 
    error_img = np.abs(A - Ak) 
    error = np.linalg.norm(A - Ak, "fro") 
 
    print(f"k = {k}, Frobenius error = {error:.4f}") 
 
    plt.figure(figsize=(12, 4)) 
 
    plt.subplot(1, 3, 1) 
    plt.imshow(A, cmap="gray") 
    plt.title("Original") 
    plt.axis("off") 
 
    plt.subplot(1, 3, 2) 
    plt.imshow(np.clip(Ak, 0, 255), cmap="gray") 
    plt.title(f"EVD, k = {k}") 
    plt.axis("off") 
 
    plt.subplot(1, 3, 3) 
    plt.imshow(error_img, cmap="gray") 
    plt.title("Error") 
    plt.axis("off") 
 
    plt.tight_layout() 
    plt.show() 
 
 
# Error for all valid k values, as required in the assignment 
 
errors = [] 
 
for k in valid_k: 
 
    Lambda = np.zeros((n, n), dtype=complex) 
    Lambda[:k, :k] = np.diag(eigval[:k]) 
 
    Ak = np.real(Q @ Lambda @ Q_inv) 
 
    errors.append( 
        np.linalg.norm(A - Ak, "fro") 
    ) 
 
plt.figure(figsize=(8, 5)) 
plt.plot(valid_k, errors, marker="o") 
plt.xlabel("k") 
plt.ylabel("Frobenius Error") 
plt.title("EVD Reconstruction Error") 
plt.grid() 
plt.show()