import cv2
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-10

# Sort eigenvalues by magnitude and keep conjugate pairs together
def sort_eigenpairs(eigval, Q):
    n = len(eigval)
    used = [False] * n
    groups = []

    for i in range(n):
        if used[i]:
            continue

        if abs(eigval[i].imag) < TOL:
            groups.append([i])
            used[i] = True
        else:
            target = np.conj(eigval[i])
            pair = next(
                (j for j in range(n)
                 if not used[j] and j != i
                 and abs(eigval[j] - target) < TOL),
                -1
            )

            if pair == -1:
                raise ValueError("Complex eigenvalue has no conjugate pair.")

            groups.append([i, pair])
            used[i] = used[pair] = True

    groups.sort(key=lambda g: abs(eigval[g[0]]), reverse=True)
    order = [i for group in groups for i in group]

    return eigval[order], Q[:, order]


# Find k values that do not split conjugate pairs
def get_valid_k(eigval):
    valid_k = []
    k = 0

    while k < len(eigval):
        k += 1

        if abs(eigval[k - 1].imag) >= TOL:
            if k >= len(eigval):
                raise ValueError("Unpaired complex eigenvalue.")
            k += 1

        valid_k.append(k)

    return valid_k


# Combine a conjugate eigenpair into a real contribution
def pair_contribution(lam, q, p):
    a, b = lam.real, lam.imag
    x, y = q.real, q.imag
    u, v = p.real, p.imag

    real_part = a * x - b * y
    imag_part = a * y + b * x

    return 2 * (
        np.outer(real_part, u) -
        np.outer(imag_part, v)
    )


# Reconstruct using the first k valid eigenpairs
def reconstruct_evd(eigval, Q, Q_inv, k, n):
    Ak = np.zeros((n, n), dtype=np.float64)
    i = 0

    while i < k:
        lam = eigval[i]

        if abs(lam.imag) < TOL:
            q = Q[:, i]
            p = Q_inv[i, :]

            if (abs(lam.imag) > TOL or
                np.max(np.abs(q.imag)) > TOL or
                np.max(np.abs(p.imag)) > TOL):
                raise ValueError("Expected a real eigenpair.")

            Ak += lam.real * np.outer(q.real, p.real)
            i += 1

        else:
            if i + 1 >= k:
                raise ValueError(
                    "k splits a complex conjugate eigenvalue pair."
                )

            if abs(eigval[i + 1] - np.conj(lam)) > TOL:
                raise ValueError("Invalid conjugate eigenvalue pair.")

            Ak += pair_contribution(
                lam, Q[:, i], Q_inv[i, :]
            )
            i += 2

    if not np.isrealobj(Ak):
        raise ValueError("Reconstruction contains complex values.")

    return Ak


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

img = cv2.imread("cat_15.jpeg")

if img is None:
    raise FileNotFoundError("Could not find 'cat_15.jpeg'.")

A = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

if A.shape[0] != A.shape[1]:
    raise ValueError("EVD requires a square image.")

n = A.shape[0]
print("Image size:", A.shape)


# ---------------------------------------------------------
# EVD: A = Q Lambda Q^(-1)
# ---------------------------------------------------------

eigval, Q = np.linalg.eig(A)
eigval, Q = sort_eigenpairs(eigval, Q)
Q_inv = np.linalg.inv(Q)


# ---------------------------------------------------------
# Select valid k values
# ---------------------------------------------------------

valid_k = get_valid_k(eigval)
chosen_k = [10, 50, 100]

ks = [
    min(valid_k, key=lambda x: abs(x - target))
    for target in chosen_k
]

print("Selected EVD k values:", ks)


# ---------------------------------------------------------
# Reconstruct selected k values
# ---------------------------------------------------------

reconstructed = {}
errors = {}

for k in ks:
    Ak = reconstruct_evd(
        eigval, Q, Q_inv, k, n
    )

    reconstructed[k] = Ak
    error = np.linalg.norm(A - Ak, "fro")
    errors[k] = error

    pixels = Ak.flatten()

    print("\n------------------------------")
    print("k =", k)
    print("------------------------------")
    print("Reconstructed dtype:", Ak.dtype)
    print("Contains complex values:", np.iscomplexobj(Ak))
    print("Is real-valued:", np.isrealobj(Ak))
    print("Flattened dtype:", pixels.dtype)
    print(
        "Flattened pixels contain complex values:",
        np.iscomplexobj(pixels)
    )
    print("Frobenius error:", f"{error:.4f}")


# ---------------------------------------------------------
# Original and reconstructed images
# ---------------------------------------------------------

plt.figure(figsize=(12, 9))

plt.subplot(2, 2, 1)
plt.imshow(A, cmap="gray")
plt.title("Original")
plt.axis("off")

for i, k in enumerate(ks):
    plt.subplot(2, 2, i + 2)
    plt.imshow(
        np.clip(reconstructed[k], 0, 255),
        cmap="gray"
    )
    plt.title(f"EVD Reconstruction, k = {k}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Reconstruction error images
# ---------------------------------------------------------

plt.figure(figsize=(12, 4))

for i, k in enumerate(ks):
    error_img = np.abs(A - reconstructed[k])

    plt.subplot(1, 3, i + 1)
    plt.imshow(error_img, cmap="gray")
    plt.title(f"Error, k = {k}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Error for all valid k values
# ---------------------------------------------------------

all_errors = []

for k in valid_k:
    Ak = reconstruct_evd(
        eigval, Q, Q_inv, k, n
    )
    all_errors.append(
        np.linalg.norm(A - Ak, "fro")
    )


# ---------------------------------------------------------
# Frobenius error vs k
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(valid_k, all_errors, marker="o")
plt.xlabel("k")
plt.ylabel("Frobenius Error")
plt.title("EVD Reconstruction Error")
plt.grid()
plt.tight_layout()
plt.show()