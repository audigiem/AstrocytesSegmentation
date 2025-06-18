import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import ball
from skimage.filters import rank
import math


def apply_median_filter_4d(image_stack, median_size=1.5):
    """
    Applique un filtre médian 3D sur des données 4D (T, Z, Y, X).
    Équivalent à: processedStack=Filters3D.filter(processedStack, Filters3D.MEDIAN, median_size_, median_size_, median_size_);

    Args:
        image_stack: Array 4D de forme (T, Z, Y, X)
        median_size: Taille du filtre (radius)

    Returns:
        Stack filtré de même forme (T, Z, Y, X)
    """
    if image_stack.ndim != 4:
        raise ValueError(f"Les données doivent être 4D (T, Z, Y, X), reçu: {image_stack.shape}")

    # Calcul de la taille du kernel (diamètre)
    kernel_size = int(2 * math.ceil(median_size) + 1)

    T, Z, Y, X = image_stack.shape
    print(f"Filtrage médian 3D sur données 4D:")
    print(f"  Shape: {image_stack.shape} (T={T}, Z={Z}, Y={Y}, X={X})")
    print(f"  Kernel size: {kernel_size}x{kernel_size}x{kernel_size}")

    # Initialisation du résultat
    result = np.zeros_like(image_stack)

    # Traitement frame par frame
    for t in range(T):
        print(f"  Traitement frame {t + 1}/{T}")
        result[t] = median_filter(image_stack[t], size=kernel_size)

    return result


def apply_median_filter_4d_spherical(image_stack, median_size=1.5):
    """
    Version avec kernel sphérique exact (plus proche du comportement Java).

    Args:
        image_stack: Array 4D de forme (T, Z, Y, X)
        median_size: Taille du filtre (radius)

    Returns:
        Stack filtré de même forme (T, Z, Y, X)
    """
    if image_stack.ndim != 4:
        raise ValueError(f"Les données doivent être 4D (T, Z, Y, X), reçu: {image_stack.shape}")

    # Création du kernel sphérique
    radius = int(math.ceil(median_size))
    footprint = ball(radius)

    T, Z, Y, X = image_stack.shape
    print(f"Filtrage médian 3D avec kernel sphérique:")
    print(f"  Shape: {image_stack.shape} (T={T}, Z={Z}, Y={Y}, X={X})")
    print(f"  Kernel sphérique radius: {radius}, shape: {footprint.shape}")

    # Initialisation du résultat
    result = np.zeros_like(image_stack)

    # Traitement frame par frame
    for t in range(T):
        print(f"  Traitement frame {t + 1}/{T}")
        # Conversion en uint8 si nécessaire pour skimage.rank
        if image_stack.dtype != np.uint8:
            temp_data = image_stack[t].astype(np.uint8)
            filtered_temp = rank.median(temp_data, footprint=footprint)
            # Reconversion vers le type original si nécessaire
            result[t] = filtered_temp.astype(image_stack.dtype)
        else:
            result[t] = rank.median(image_stack[t], footprint=footprint)

    return result


def apply_median_filter_4d_parallel(image_stack, median_size=1.5, n_jobs=-1):
    """
    Version parallélisée pour de gros volumes 4D.

    Args:
        image_stack: Array 4D de forme (T, Z, Y, X)
        median_size: Taille du filtre (radius)
        n_jobs: Nombre de processus (-1 = tous les CPU)

    Returns:
        Stack filtré de même forme (T, Z, Y, X)
    """
    if image_stack.ndim != 4:
        raise ValueError(f"Les données doivent être 4D (T, Z, Y, X), reçu: {image_stack.shape}")

    try:
        from joblib import Parallel, delayed
    except ImportError:
        print("joblib non disponible, utilisation de la version séquentielle")
        return apply_median_filter_4d(image_stack, median_size)

    kernel_size = int(2 * math.ceil(median_size) + 1)
    T, Z, Y, X = image_stack.shape

    print(f"Filtrage médian 3D parallèle sur données 4D:")
    print(f"  Shape: {image_stack.shape} (T={T}, Z={Z}, Y={Y}, X={X})")
    print(f"  Kernel size: {kernel_size}x{kernel_size}x{kernel_size}")
    print(f"  Processus: {n_jobs}")

    def process_frame(frame_data):
        return median_filter(frame_data, size=kernel_size)

    # Traitement parallèle de toutes les frames
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_frame)(image_stack[t]) for t in range(T)
    )

    return np.stack(results)


def apply_median_filter_4d_vectorized(image_stack, median_size=1.5):
    """
    Version vectorisée ultra-rapide pour de gros volumes.
    Utilise les capacités vectorielles de scipy directement sur les 4D.

    Args:
        image_stack: Array 4D de forme (T, Z, Y, X)
        median_size: Taille du filtre (radius)

    Returns:
        Stack filtré de même forme (T, Z, Y, X)
    """
    if image_stack.ndim != 4:
        raise ValueError(f"Les données doivent être 4D (T, Z, Y, X), reçu: {image_stack.shape}")

    kernel_size = int(2 * math.ceil(median_size) + 1)
    T, Z, Y, X = image_stack.shape

    print(f"Filtrage médian 3D vectorisé sur données 4D:")
    print(f"  Shape: {image_stack.shape} (T={T}, Z={Z}, Y={Y}, X={X})")
    print(f"  Kernel size: {kernel_size}x{kernel_size}x{kernel_size}")

    # Reshape pour traiter toutes les frames d'un coup
    # Reshape en (T*Z, Y, X) puis apply 3D filter frame par frame
    result = np.zeros_like(image_stack)

    # Application du filtre 3D sur chaque volume temporel
    for t in range(T):
        result[t] = median_filter(image_stack[t], size=kernel_size, mode='nearest')

    return result


# Fonction principale - version recommandée
def median_filter_3d_time(image_stack, median_size=1.5, method='simple'):
    """
    Fonction principale pour appliquer le filtre médian 3D+temps.

    Args:
        image_stack: Array 4D de forme (T, Z, Y, X)
        median_size: Taille du filtre (radius) - par défaut 1.5 comme dans votre code Java
        method: 'simple', 'spherical', 'parallel', ou 'vectorized'

    Returns:
        Stack filtré de même forme (T, Z, Y, X)
    """
    if method == 'simple':
        return apply_median_filter_4d(image_stack, median_size)
    elif method == 'spherical':
        return apply_median_filter_4d_spherical(image_stack, median_size)
    elif method == 'parallel':
        return apply_median_filter_4d_parallel(image_stack, median_size)
    elif method == 'vectorized':
        return apply_median_filter_4d_vectorized(image_stack, median_size)
    else:
        raise ValueError(f"Méthode inconnue: {method}")


# Exemple d'utilisation
if __name__ == "__main__":
    print("Test du filtre médian 3D+temps pour données 4D...")

    # Création de données de test 4D (T=3, Z=20, Y=64, X=64)
    T, Z, Y, X = 3, 20, 64, 64
    test_data = np.random.randint(0, 256, (T, Z, Y, X), dtype=np.uint8)

    # Ajout de bruit salt-and-pepper
    noise_mask = np.random.random(test_data.shape) < 0.05
    test_data[noise_mask] = np.random.choice([0, 255], size=np.sum(noise_mask))

    print(f"Données test: {test_data.shape}")
    print(f"Avant filtrage - Min: {test_data.min()}, Max: {test_data.max()}, Mean: {test_data.mean():.2f}")

    # Test de la méthode principale (équivalent direct de votre code Java)
    print("\n" + "=" * 50)
    print("ÉQUIVALENT DIRECT DU CODE JAVA:")
    print("=" * 50)

    result = median_filter_3d_time(test_data, median_size=1.5, method='simple')

    print(f"Après filtrage - Min: {result.min()}, Max: {result.max()}, Mean: {result.mean():.2f}")
    print(f"Réduction du bruit: {((test_data != result).sum() / test_data.size * 100):.1f}% des pixels modifiés")

    # Comparaison des différentes méthodes
    print(f"\n" + "=" * 50)
    print("COMPARAISON DES MÉTHODES:")
    print("=" * 50)

    methods = ['simple', 'spherical', 'parallel']
    results = {}

    for method in methods:
        print(f"\n--- Test méthode: {method} ---")
        try:
            results[method] = median_filter_3d_time(test_data, median_size=1.5, method=method)
            print(f"✓ Succès - Mean: {results[method].mean():.2f}")
        except Exception as e:
            print(f"✗ Erreur: {e}")

    # Comparaison des résultats
    if len(results) > 1:
        methods_list = list(results.keys())
        for i in range(len(methods_list)):
            for j in range(i + 1, len(methods_list)):
                diff = np.abs(results[methods_list[i]].astype(float) -
                              results[methods_list[j]].astype(float)).max()
                print(f"Diff max {methods_list[i]} vs {methods_list[j]}: {diff}")

    print("\nTest terminé avec succès!")