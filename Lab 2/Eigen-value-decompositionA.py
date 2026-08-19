import cv2
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-10

def sort_eigenpairs(eigval, Q):
    """Sort eigenpairs by descending eigenvalue magnitude."""
    n = len(eigval)
    used = [False] * n
    groups = []

    # Keep complex conjugate pairs together
    for i in range(n):
        if used[i]:
            continue

        if abs(eigval[i].imag) < 1e-10:
            groups.append([i])
            used[i] = True
        else:
            target = np.conj(eigval[i])
            pair = -1

            for j in range(n):
                if not used[j] and j != i:
                    if abs(eigval[j] - target) < 1e-8:
                        pair = j
                        break

            if pair == -1:
                raise ValueError("Complex eigenvalue has no conjugate pair.")

            groups.append([i, pair])
            used[i] = used[pair] = True

    # Selection sort by magnitude
    for i in range(len(groups)):
        max_pos = i
        max_mag = abs(eigval[groups[i][0]])

        for j in range(i + 1, len(groups)):
            mag = abs(eigval[groups[j][0]])

            if mag > max_mag:
                max_mag = mag
                max_pos = j

        if max_pos != i:
            groups[i], groups[max_pos] = groups[max_pos], groups[i]

    order = [idx for group in groups for idx in group]

    return eigval[order], Q[:, order]


def get_valid_k(eigval):
    """Return k values that do not split conjugate pairs."""
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


def pair_contribution(lam, q, p):
    """Return the real contribution of a conjugate eigenvalue pair."""
    a, b = lam.real, lam.imag

    x, y = q.real, q.imag
    u, v = p.real, p.imag

    real_part = a * x - b * y
    imag_part = b * x + a * y

    return 2.0 * (
        np.outer(real_part, u) -
        np.outer(imag_part, v)
    )


def reconstruct_evd(eigval, Q, Q_inv, k, n):
    """Reconstruct A using the first k eigenpairs."""
    Ak = np.zeros((n, n), dtype=np.float64)
    i = 0

    while i < k:

        if abs(eigval[i].imag) < TOL:
            # Real eigenvalue contribution
            lam = eigval[i].real
            Ak += lam * np.outer(
                Q[:, i].real,
                Q_inv[i, :].real
            )
            i += 1

        else:
            # Complex eigenvalues must be taken as a pair
            if i + 1 >= k:
                raise ValueError(
                    "k splits a complex conjugate eigenvalue pair."
                )

            Ak += pair_contribution(
                eigval[i],
                Q[:, i],
                Q_inv[i, :]
            )
            i += 2

    # Ak is created as float64, so no complex values are introduced.
    if np.iscomplexobj(Ak):
        raise ValueError("Reconstruction contains complex values.")

    return Ak


# Load image
img = cv2.imread("cat_15.jpeg")

A = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

if A.shape[0] != A.shape[1]:
    raise ValueError("EVD requires a square image.")

n = A.shape[0]
print("Image size:", A.shape)


# EVD: A = Q Lambda Q^(-1)
eigval, Q = np.linalg.eig(A)

eigval, Q = sort_eigenpairs(eigval, Q)

Q_inv = np.linalg.inv(Q)

valid_k = get_valid_k(eigval)


# Select the closest valid k to 10, 50 and 100
chosen_k = [10, 50, 100]

ks = [
    min(valid_k, key=lambda x: abs(x - target))
    for target in chosen_k
]

print("EVD k values:", ks)


# Reconstruct selected k values
reconstructed = {}

for k in ks:
    Ak = reconstruct_evd(eigval, Q, Q_inv, k, n)
    reconstructed[k] = Ak

    error = np.linalg.norm(A - Ak, "fro")

    print(f"\nk = {k}")
    print("Reconstructed dtype:", Ak.dtype)
    print("Contains complex values:", np.iscomplexobj(Ak))
    print("Is real-valued:", np.isrealobj(Ak))
    print("Frobenius error:", f"{error:.4f}")

    # Check the flattened image too
    pixels = Ak.flatten()
    print("Flattened dtype:", pixels.dtype)
    print(
        "Flattened pixels contain complex values:",
        np.iscomplexobj(pixels)
    )

    # Display original, reconstruction and error
    error_img = np.abs(A - Ak)

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


# Reconstruction error for all valid k
errors = []

for k in valid_k:
    Ak = reconstruct_evd(eigval, Q, Q_inv, k, n)
    errors.append(np.linalg.norm(A - Ak, "fro"))


plt.figure(figsize=(8, 5))
plt.plot(valid_k, errors, marker="o")
plt.xlabel("k")
plt.ylabel("Frobenius Error")
plt.title("EVD Reconstruction Error")
plt.grid()
plt.show()