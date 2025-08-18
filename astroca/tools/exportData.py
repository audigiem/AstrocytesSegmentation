"""
@file exportData.py
@brief This module provides functionality to export 3D image sequences with time dimension to various formats.
@detail It includes methods to save the data either as one .tif file that contains all time frames or as multiple .tif files, one for each time frame in a directory.
"""

import os
import numpy as np
from tifffile import imwrite
import torch
from typing import Optional, Union
import threading
from tqdm import tqdm


def export_data(
    data: np.ndarray,
    output_path: str,
    export_as_single_tif: bool = True,
    file_name: str = "exported_sequence",
    directory_name: str = "exported_data",
):
    """
    Export 3D image sequence data to .tif files, readable as T-Z-X-Y in FIJI.

    @param data: 4D numpy array of shape (T, Z, Y, X) representing the image sequence.
    @param output_path: Path where the exported files will be saved.
    @param export_as_single_tif: If True, saves all time frames in one .tif file; otherwise saves each time frame as a separate .tif file.
    @param file_name: Name of the file to save if exporting as a single .tif file.
    @param directory_name: Name of the directory to save the exported files if exporting multiple .tif files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if export_as_single_tif:
        T, Z, Y, X = data.shape

        # Reshape into a 5D array for ImageJ: (T, Z, C=1, Y, X)
        data_5d = data[:, :, np.newaxis, :, :]  # (T, Z, 1, Y, X)
        if not file_name.endswith(".tif"):
            file_name += ".tif"

        imwrite(
            os.path.join(output_path, file_name),
            data_5d,
            imagej=True,
            metadata={"axes": "TZCYX", "Frames": T, "Slices": Z, "Channels": 1},
        )
        print(f"Exported all time frames to {os.path.join(output_path, file_name)}")
    else:
        # Export each time frame separately (no ImageJ metadata needed)
        directory_path = os.path.join(output_path, directory_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for t in range(data.shape[0]):
            imwrite(os.path.join(directory_path, f"time_frame_{t}.tif"), data[t])
            print(
                f"Exported time frame {t} to {os.path.join(directory_path, f'time_frame_{t}.tif')}"
            )


def save_numpy_tab(
    data: np.ndarray, output_path: str, file_name: str = "exported_data.npy"
):
    """
    Save a numpy array to a .npy file.

    @param data: Numpy array to save.
    @param output_path: Path where the .npy file will be saved.
    @param file_name: Name of the .npy file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(os.path.join(output_path, file_name), data)
    print(f"Saved numpy array to {os.path.join(output_path, file_name)}")


def export_data_GPU(
    data: torch.Tensor,
    output_path: str,
    export_as_single_tif: bool = True,
    file_name: str = "exported_sequence",
    directory_name: str = "exported_data",
    async_export: bool = True,
    chunk_size: int = 5,
):
    """
    Export GPU tensor to .tif files avec transfert optimisé

    @param data: 4D tensor PyTorch (T, Z, Y, X) sur GPU
    @param output_path: Chemin de sauvegarde
    @param export_as_single_tif: Si True, un seul .tif ; sinon un .tif par frame
    @param file_name: Nom du fichier pour export unique
    @param directory_name: Nom du dossier pour exports multiples
    @param async_export: Export asynchrone pour ne pas bloquer le GPU
    @param chunk_size: Taille des chunks pour transfert optimisé
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # print(f" - [GPU Export] Starting export of tensor shape {data.shape}")

    if export_as_single_tif:
        if async_export:
            return export_single_tif_GPU_async(data, output_path, file_name, chunk_size)
        else:
            export_single_tif_GPU_sync(data, output_path, file_name, chunk_size)
    else:
        if async_export:
            return export_multiple_tifs_GPU_async(
                data, output_path, directory_name, chunk_size
            )
        else:
            export_multiple_tifs_GPU_sync(data, output_path, directory_name, chunk_size)


def export_single_tif_GPU_sync(
    data: torch.Tensor, output_path: str, file_name: str, chunk_size: int = 5
):
    """
    Export synchrone d'un tensor GPU vers un fichier .tif unique
    """
    T, Z, Y, X = data.shape

    if not file_name.endswith(".tif"):
        file_name += ".tif"

    print(f" - [GPU Export] Transferring {T} frames to CPU by chunks of {chunk_size}")

    # Pré-allocation CPU pour éviter les réallocations
    data_cpu = np.empty((T, Z, Y, X), dtype=data.cpu().numpy().dtype)

    # Transfert par chunks pour éviter l'OOM
    # for t_start in tqdm(range(0, T, chunk_size), desc="GPU→CPU transfer", unit="chunk"):
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)

        # Transfert synchrone du chunk
        with torch.cuda.stream(torch.cuda.default_stream()):
            chunk = data[t_start:t_end].cpu().numpy()
            data_cpu[t_start:t_end] = chunk

        # Nettoyage explicite
        del chunk
        torch.cuda.empty_cache()

    # Reshape pour ImageJ : (T, Z, C=1, Y, X)
    data_5d = data_cpu[:, :, np.newaxis, :, :]

    # Sauvegarde avec métadonnées ImageJ
    imwrite(
        os.path.join(output_path, file_name),
        data_5d,
        imagej=True,
        metadata={"axes": "TZCYX", "fps": 30.0},
    )

    print(f" - [GPU Export] Exported to {os.path.join(output_path, file_name)}")
    del data_cpu, data_5d


def export_single_tif_GPU_async(
    data: torch.Tensor, output_path: str, file_name: str, chunk_size: int = 5
) -> threading.Thread:
    """
    Export asynchrone pour ne pas bloquer le processing GPU
    """

    def async_worker():
        export_single_tif_GPU_sync(data, output_path, file_name, chunk_size)

    thread = threading.Thread(target=async_worker, daemon=True)
    thread.start()

    # print(f" - [GPU Export] Async export started for {file_name}")
    return thread


def export_multiple_tifs_GPU_sync(
    data: torch.Tensor, output_path: str, directory_name: str, chunk_size: int = 5
):
    """
    Export synchrone vers fichiers .tif multiples
    """
    T, Z, Y, X = data.shape
    directory_path = os.path.join(output_path, directory_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # print(f" - [GPU Export] Exporting {T} frames to separate .tif files")

    # Export par chunks pour optimiser les transferts
    for t_start in tqdm(range(0, T, chunk_size), desc="Exporting frames", unit="chunk"):
        t_end = min(t_start + chunk_size, T)

        # Transfert du chunk vers CPU
        chunk_cpu = data[t_start:t_end].cpu().numpy()

        # Sauvegarde de chaque frame du chunk
        for i, t in enumerate(range(t_start, t_end)):
            frame_data = chunk_cpu[i]  # Shape: (Z, Y, X)
            file_path = os.path.join(directory_path, f"frame_{t:04d}.tif")

            imwrite(file_path, frame_data, photometric="minisblack")

        # Nettoyage
        del chunk_cpu
        torch.cuda.empty_cache()

    print(f" - [GPU Export] Exported {T} frames to {directory_path}")


def export_multiple_tifs_GPU_async(
    data: torch.Tensor, output_path: str, directory_name: str, chunk_size: int = 5
) -> threading.Thread:
    """
    Export asynchrone vers fichiers multiples
    """

    def async_worker():
        export_multiple_tifs_GPU_sync(data, output_path, directory_name, chunk_size)

    thread = threading.Thread(target=async_worker, daemon=True)
    thread.start()

    # print(f" - [GPU Export] Async export started to {directory_name}/")
    return thread


def save_tensor_as_numpy_GPU(
    data: torch.Tensor,
    output_path: str,
    file_name: str = "exported_data.npy",
    async_save: bool = True,
) -> Optional[threading.Thread]:
    """
    Sauvegarde d'un tensor GPU vers .npy avec optimisations
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def save_worker():
        # print(f" - [GPU Export] Transferring tensor to CPU for .npy save...")
        data_cpu = data.cpu().numpy()
        file_path = os.path.join(output_path, file_name)
        np.save(file_path, data_cpu)
        print(f" - [GPU Export] Saved tensor to {file_path}")
        del data_cpu

    if async_save:
        thread = threading.Thread(target=save_worker, daemon=True)
        thread.start()
        return thread
    else:
        save_worker()
        return None


def export_data_GPU_with_memory_optimization(
    data: torch.Tensor,
    output_path: str,
    export_as_single_tif: bool = True,
    file_name: str = "exported_sequence",
    directory_name: str = "exported_data",
    max_memory_usage_mb: float = 2048.0,  # 2GB par défaut
):
    """
    Export avec optimisation mémoire automatique
    """
    T, Z, Y, X = data.shape
    element_size_mb = data.element_size() / 1024**2

    # Calcul du chunk size optimal basé sur la mémoire disponible
    memory_per_frame = Z * Y * X * element_size_mb
    optimal_chunk_size = max(1, int(max_memory_usage_mb / memory_per_frame))

    # print(
    #     f" - [GPU Export] Auto chunk size: {optimal_chunk_size} frames "
    #     f"(memory per frame: {memory_per_frame:.1f}MB)"
    # )

    return export_data_GPU(
        data,
        output_path,
        export_as_single_tif,
        file_name,
        directory_name,
        async_export=True,
        chunk_size=optimal_chunk_size,
    )


def export_data_GPU_streaming(
    data: torch.Tensor, output_path: str, file_name: str = "streamed_export.tif"
):
    """
    Export streaming pour très gros datasets - écrit frame par frame
    """
    T, Z, Y, X = data.shape

    if not file_name.endswith(".tif"):
        file_name += ".tif"

    file_path = os.path.join(output_path, file_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # print(f" - [GPU Export] Streaming export of {T} frames...")

    # Export frame par frame pour minimiser l'usage mémoire
    for t in tqdm(range(T), desc="Streaming export", unit="frame"):
        frame_gpu = data[t]  # (Z, Y, X)
        frame_cpu = frame_gpu.cpu().numpy()

        if t == 0:
            # Créer le fichier avec le premier frame
            imwrite(file_path, frame_cpu, photometric="minisblack")
        else:
            # Append les frames suivants
            imwrite(file_path, frame_cpu, photometric="minisblack", append=True)

        del frame_cpu

        # Nettoyage périodique
        if t % 10 == 0:
            torch.cuda.empty_cache()

    print(f" - [GPU Export] Streaming export completed: {file_path}")


def batch_export_GPU_tensors(
    tensor_list: list[torch.Tensor],
    output_path: str,
    file_names: list[str],
    export_as_single_tif: bool = True,
    max_concurrent_exports: int = 3,
):
    """
    Export par batch de plusieurs tenseurs avec contrôle de concurrence
    """
    if len(tensor_list) != len(file_names):
        raise ValueError("Number of tensors must match number of file names")

    # print(f" - [GPU Export] Batch export of {len(tensor_list)} tensors...")

    # Limiter le nombre d'exports concurrents
    from concurrent.futures import ThreadPoolExecutor

    def export_single_tensor(args):
        tensor, name = args
        return export_data_GPU(
            tensor, output_path, export_as_single_tif, name, async_export=False
        )

    with ThreadPoolExecutor(max_workers=max_concurrent_exports) as executor:
        futures = [
            executor.submit(export_single_tensor, (tensor, name))
            for tensor, name in zip(tensor_list, file_names)
        ]

        # Attendre la completion
        for future in tqdm(futures, desc="Batch export", unit="tensor"):
            future.result()

    print(f" - [GPU Export] Batch export completed")
