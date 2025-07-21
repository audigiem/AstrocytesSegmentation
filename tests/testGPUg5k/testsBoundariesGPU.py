from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
from astroca.tools.loadData import load_data, read_config
from astroca.croppingBoundaries.cropper import crop_boundaries
import cupy as cp

def test_boundariesGPU(file_path: str = "home/maudigie/data/inputData/20stepsTimeScene.tif") -> None:
    """
    Test the boundariesGPU function.
    """
    # determine if GPU is available
    GPU_AVAILABLE = cp.cuda.is_available()
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU is not available. Please check your CUDA installation.")
    print(f"GPU available: {GPU_AVAILABLE}")
    # Load configuration parameters
    data = load_data(file_path, GPU_AVAILABLE=GPU_AVAILABLE)
    params = read_config()
    params["GPU_AVAILABLE"] = GPU_AVAILABLE
    index_xmin, index_xmax, default_value, data = compute_boundaries(crop_boundaries(data, params), params)

if __name__ == "__main__":
    test_boundariesGPU()