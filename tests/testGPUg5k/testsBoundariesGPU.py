from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.tools.loadData import load_data, read_config
from astroca.croppingBoundaries.cropper import crop_boundaries
import torch


def test_boundariesGPU(
    file_path: str = "/home/maudigie/data/inputData/20stepsTimeScene.tif",
) -> None:
    """
    Test the boundaries function using GPU if available.
    """
    # Détermine si un GPU est disponible via torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU is not available. Please check your CUDA installation.")

    print(f"GPU available: {GPU_AVAILABLE}")

    # Chargement des données (format torch.Tensor si GPU)
    data = load_data(file_path, GPU_AVAILABLE=GPU_AVAILABLE)

    # Chargement des paramètres depuis le fichier de configuration
    params = read_config()
    params["GPU_AVAILABLE"] = 1 if GPU_AVAILABLE else 0

    # Application du cropping
    cropped_data = crop_boundaries(data, params)

    # Calcul des bornes utiles sur les données rognées
    index_xmin, index_xmax, default_value, data = compute_boundaries(
        cropped_data, params
    )


if __name__ == "__main__":
    test_boundariesGPU()
