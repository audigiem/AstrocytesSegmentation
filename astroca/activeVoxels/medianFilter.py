import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.ndimage import median_filter
from numba import njit, prange




def unified_median_filter_3d(
        data: np.ndarray,
        radius: float = 1.5,
        border_mode: str = 'reflect',
        n_workers: int = None
) -> np.ndarray:
    """
    Median filter 3D unifié pour stacks 4D (T,Z,Y,X)

    Args:
        data: Input stack (T,Z,Y,X)
        radius: Rayon de la sphère (1.5 → voisinage 3×3×7)
        border_mode: 'reflect', 'nearest', 'constant', etc.
        n_workers: Nombre de threads
    """
    print(f" - Apply 3D median filter with radius={radius}, border mode='{border_mode}'")
    if border_mode == 'ignore':
        T, Z, Y, X = data.shape
        data_3D = data.reshape(T * Z, Y, X)  # Reshape pour traiter comme 3D
        offsets = generate_spherical_offsets(radius)
        median_filtered = apply_median_filter_3d_ignore_border(data_3D, offsets)
        data_filtered_4D = median_filtered.reshape(T, Z, Y, X)
        return data_filtered_4D
    print(f" - Apply 3D median filter with radius={radius}, border mode='{border_mode}'")
    r = int(np.ceil(radius))

    # Créer le masque sphérique
    shape = (2 * r + 1, 2 * r + 1, 2 * r + 1)
    mask = np.zeros(shape, dtype=bool)
    center = np.array([r, r, r])
    for idx in np.ndindex(shape):
        if np.linalg.norm(np.array(idx) - center) <= radius:
            mask[idx] = True

    # Padding manuel : seulement sur les axes Z, Y, X
    pad_width = [(0, 0), (r, r), (r, r), (r, r)]
    padded = np.pad(data, pad_width=pad_width, mode=border_mode)

    filtered = np.empty_like(data)

    def process_frame(t):
        result = median_filter(
            padded[t], footprint=mask, mode=border_mode
        )
        # Enlever le padding
        filtered[t] = result[r:-r, r:-r, r:-r]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(
            executor.map(process_frame, range(data.shape[0])),
            total=data.shape[0], desc=f"Processing frames with median filter and {border_mode} border condition",
            unit="frame"
        ))

    return filtered


@njit
def quickselect_median(arr, n):
    """
    Calcul optimisé de la médiane pour de petits tableaux
    Utilise quickselect pour trouver l'élément médian sans tri complet
    """
    if n == 0:
        return 0
    if n == 1:
        return arr[0]

    # Pour les petits tableaux, tri par insertion est plus rapide
    if n <= 20:
        # Tri par insertion
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

        if n % 2 == 1:
            return arr[n // 2]
        else:
            return (arr[n // 2 - 1] + arr[n // 2]) / 2.0

    # Pour les tableaux plus grands, utiliser quickselect
    # Mais en pratique, nos voisinages sphériques sont petits
    return quickselect_kth(arr, n, n // 2)


@njit
def quickselect_kth(arr, n, k):
    """
    Trouve le k-ième élément le plus petit (0-indexé)
    """
    left = 0
    right = n - 1

    while left < right:
        pivot_idx = partition(arr, left, right)
        if pivot_idx == k:
            return arr[k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1

    return arr[k]


@njit
def partition(arr, left, right):
    """
    Partition pour quickselect
    """
    pivot = arr[right]
    i = left - 1

    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


@njit(parallel=True)
def apply_median_filter_3d_ignore_border(frame: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Applique un filtre médian 3D en ignorant les bords
    Version optimisée avec Numba
    """
    Z, Y, X = frame.shape
    result = np.empty((Z, Y, X), dtype=frame.dtype)
    max_neighbors = offsets.shape[0]

    # for z in range(Z):
    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                # Tableau temporaire pour chaque thread
                tmp_values = np.empty(max_neighbors, dtype=frame.dtype)
                count = 0

                # Collecter les valeurs valides dans le voisinage sphérique
                for i in range(offsets.shape[0]):
                    dz, dy, dx = offsets[i]
                    zz, yy, xx = z + dz, y + dy, x + dx

                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        tmp_values[count] = frame[zz, yy, xx]
                        count += 1

                # Calculer la médiane sur les valeurs valides
                if count > 0:
                    result[z, y, x] = quickselect_median(tmp_values, count)
                else:
                    # Fallback : garder la valeur originale si aucun voisin valide
                    result[z, y, x] = frame[z, y, x]

    return result


def generate_spherical_offsets(radius: float):
    """
    Génère les offsets pour une sphère de rayon donné
    Utilise l'algorithme exact du code Java avec ellipsoïde normalisée
    """
    radx = rady = radz = radius  # Sphère = ellipsoïde avec rayons égaux

    vx = int(np.ceil(radx))
    vy = int(np.ceil(rady))
    vz = int(np.ceil(radz))

    # Calcul des inverses des rayons au carré (comme dans le Java)
    rx2 = 1.0 / (radx * radx) if radx != 0.0 else 0.0
    ry2 = 1.0 / (rady * rady) if rady != 0.0 else 0.0
    rz2 = 1.0 / (radz * radz) if radz != 0.0 else 0.0

    offsets = []

    # Boucles dans le même ordre que Java : k(z), j(y), i(x)
    for k in range(-vz, vz + 1):  # dz
        for j in range(-vy, vy + 1):  # dy
            for i in range(-vx, vx + 1):  # dx
                # Distance normalisée exacte du Java
                dist = (i * i) * rx2 + (j * j) * ry2 + (k * k) * rz2

                if dist <= 1.0:  # Condition exacte du Java
                    offsets.append((k, j, i))  # (dz, dy, dx)

    offsets_array = np.array(offsets, dtype=np.int32)
    # print(f"Generated {len(offsets)} offsets for radius {radius}")
    # print(f"Integer bounds: vx={vx}, vy={vy}, vz={vz}")
    # print(f"Normalization factors: rx2={rx2:.6f}, ry2={ry2:.6f}, rz2={rz2:.6f}")
    # print(f"Some offsets: {offsets_array[:10]}")  # Afficher les premiers offsets
    return offsets_array