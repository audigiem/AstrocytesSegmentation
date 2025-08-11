from astroca.varianceStabilization.varianceStabilization import (
    compute_variance_stabilization,
)
from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block
from astroca.dynamicImage.dynamicImage import compute_dynamic_image
from astroca.tools.loadData import load_data, read_config
import numpy as np
import torch


def test_boundariesGPU(
    file_path: str = "/home/maudigie/data/outputData/testGPU/bounded_image_sequence.tif",
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
    index_xmin = np.load("/home/maudigie/data/outputData/testGPU/index_Xmin.npy")
    index_xmax = np.load("/home/maudigie/data/outputData/testGPU/index_Xmax.npy")

    # Chargement des paramètres depuis le fichier de configuration
    params = read_config()
    params["GPU_AVAILABLE"] = 1 if GPU_AVAILABLE else 0

    anscombe_data = compute_variance_stabilization(data, index_xmin, index_xmax, params)

    T, Z, Y, X = anscombe_data.shape

    F0 = background_estimation_single_block(
        anscombe_data, index_xmin, index_xmax, params
    )
    dF, mean_noise = compute_dynamic_image(
        anscombe_data, F0, index_xmin, index_xmax, T, params
    )


if __name__ == "__main__":
    test_boundariesGPU()
