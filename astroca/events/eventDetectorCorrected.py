import numpy as np
import numba as nb
from numba import njit, prange, types
from typing import List, Tuple, Optional, Any
import time
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import deque
import concurrent.futures
from functools import partial
from joblib import Parallel, delayed


# Compilation Numba pour les opérations critiques
@njit(parallel=True, fastmath=True, cache=True)
def compute_correlation_batch(patterns1, patterns2, min_len=3):
    """Calcul vectorisé des corrélations croisées normalisées"""
    n_patterns = patterns1.shape[0]
    correlations = np.zeros(n_patterns, dtype=np.float32)
    
    for i in prange(n_patterns):
        p1 = patterns1[i]
        p2 = patterns2[i]
        
        # Ignorer les patterns trop courts
        if len(p1) < min_len or len(p2) < min_len:
            continue
            
        # Corrélation croisée normalisée simplifiée
        mean1 = np.mean(p1)
        mean2 = np.mean(p2)
        
        num = np.sum((p1 - mean1) * (p2 - mean2))
        den1 = np.sqrt(np.sum((p1 - mean1) ** 2))
        den2 = np.sqrt(np.sum((p2 - mean2) ** 2))
        
        if den1 > 0 and den2 > 0:
            correlations[i] = num / (den1 * den2)
    
    return correlations


@njit(parallel=True, fastmath=True, cache=True)
def find_seeds_vectorized(frame_data, processed_mask):
    """Trouve tous les seeds potentiels dans une frame"""
    h, w = frame_data.shape
    seeds = []
    values = []
    
    for z in prange(h):
        for y in range(w):
            if frame_data[z, y] > 0 and not processed_mask[z, y]:
                seeds.append((z, y))
                values.append(frame_data[z, y])
    
    return seeds, values


@njit(fastmath=True, cache=True)
def get_pattern_bounds(intensity_profile, t_start):
    """Trouve les bornes du pattern de manière optimisée"""
    n = len(intensity_profile)
    
    # Trouve le début
    start = t_start
    while start > 0 and intensity_profile[start - 1] != 0:
        start -= 1
    
    # Trouve la fin
    end = t_start
    while end < n - 1 and intensity_profile[end + 1] != 0:
        end += 1
    
    return start, end + 1


@njit(parallel=True, fastmath=True, cache=True)
def compute_3d_distances(coords1, coords2):
    """Calcule les distances 3D entre deux ensembles de coordonnées"""
    n1, n2 = len(coords1), len(coords2)
    distances = np.full((n1, n2), np.inf, dtype=np.float32)
    
    for i in prange(n1):
        for j in range(n2):
            dx = coords1[i][0] - coords2[j][0]
            dy = coords1[i][1] - coords2[j][1]
            dz = coords1[i][2] - coords2[j][2]
            distances[i, j] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    return distances


class EventDetectorTurbo:
    """
    Détecteur d'événements calciques ultra-rapide avec optimisations avancées
    """
    
    def __init__(self, av_data: np.ndarray, 
                 threshold_size_3d: int = 10,
                 threshold_size_3d_removed: int = 5, 
                 threshold_corr: float = 0.5,
                 use_gpu: bool = False,
                 n_jobs: int = -1):
        
        print("=== EventDetector Turbo - Initialisation ===")
        print(f"Données: {av_data.shape}, type: {av_data.dtype}")
        print(f"Voxels non-nuls: {np.count_nonzero(av_data):,}/{av_data.size:,}")
        
        self.av_ = av_data.astype(np.float32)
        self.time_length_, self.depth_, self.height_, self.width_ = av_data.shape
        
        self.threshold_size_3d_ = threshold_size_3d
        self.threshold_size_3d_removed_ = threshold_size_3d_removed
        self.threshold_corr_ = threshold_corr
        self.use_gpu_ = use_gpu and self._check_gpu()
        self.n_jobs_ = n_jobs
        
        # Structures optimisées
        self.id_connected_voxel_ = np.zeros_like(self.av_, dtype=np.int32)
        self.final_id_events_ = []
        
        # Pré-calcul des masques et indices
        self._precompute_structures()
        
        print(f"GPU activé: {self.use_gpu_}")
        print(f"Jobs parallèles: {self.n_jobs_}")
    
    def _check_gpu(self):
        """Vérifie la disponibilité du GPU"""
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False
    
    def _precompute_structures(self):
        """Pré-calcule les structures pour l'optimisation"""
        print("Pré-calcul des structures...")
        
        # Masque des voxels non-nuls par frame
        self.nonzero_masks_ = []
        self.nonzero_coords_ = []
        
        for t in range(self.time_length_):
            mask = self.av_[t] != 0
            coords = np.where(mask)
            self.nonzero_masks_.append(mask)
            self.nonzero_coords_.append(coords)
        
        # Voisinage 3D pré-calculé
        self.neighbors_3d_ = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    self.neighbors_3d_.append((dz, dy, dx))
        
        print(f"Structures pré-calculées: {len(self.neighbors_3d_)} voisins 3D")
    
    def find_events(self) -> None:
        """Méthode principale optimisée pour la détection d'événements"""
        print(f"Seuils: taille={self.threshold_size_3d_}, supprimé={self.threshold_size_3d_removed_}, corr={self.threshold_corr_}")
        
        start_time = time.time()
        
        # Étape 1: Détection rapide des composantes connexes par frame
        print("Étape 1: Détection des composantes connexes...")
        frame_components = self._detect_frame_components()
        
        # Étape 2: Liaison temporelle des composantes
        print("Étape 2: Liaison temporelle...")
        temporal_events = self._link_temporal_components(frame_components)
        
        # Étape 3: Filtrage par taille et corrélation
        print("Étape 3: Filtrage et validation...")
        self._filter_and_validate_events(temporal_events)
        
        # Étape 4: Post-traitement des petites régions
        print("Étape 4: Post-traitement...")
        self._post_process_small_regions()
        
        print(f"Événements détectés: {len(self.final_id_events_)}")
        print(f"Temps total: {time.time() - start_time:.2f}s")
    
    def _detect_frame_components(self):
        """Détection rapide des composantes connexes par frame"""
        frame_components = []

        def process_frame(t):
            components = []
            av_t = self.av_[t]  # (Z, Y, X)

            for z in range(av_t.shape[0]):
                plane = av_t[z]  # (Y, X)
                if np.count_nonzero(plane) == 0:
                    continue

                labeled = label(plane > 0, connectivity=2)

                for region in regionprops(labeled):
                    coords = region.coords  # (N, 2)
                    intensities = plane[coords[:, 0], coords[:, 1]]  # (N,)

                    if len(intensities) < 3:
                        continue

                    total_intensity = np.sum(intensities)
                    if total_intensity > 0:
                        centroid_yx = np.average(coords, axis=0, weights=intensities)
                        centroid = (z, centroid_yx[0], centroid_yx[1])

                        components.append({
                            'coords': np.column_stack((np.full(len(coords), z), coords)),  # (N, 3): (z, y, x)
                            'centroid': centroid,
                            'intensity': total_intensity,
                            'max_intensity': np.max(intensities),
                            'area': len(coords)
                        })

            return components

        # Traitement parallèle
        if self.n_jobs_ != 1:
            frame_components = Parallel(n_jobs=self.n_jobs_)(
                delayed(process_frame)(t) for t in tqdm(range(self.time_length_))
            )
        else:
            frame_components = [process_frame(t) for t in tqdm(range(self.time_length_))]

        return frame_components

    
    def _link_temporal_components(self, frame_components):
        """Liaison temporelle des composantes avec optimisation"""
        temporal_events = []
        event_id = 1
        
        # Suivi des événements actifs
        active_events = {}
        
        for t in tqdm(range(self.time_length_), desc="Liaison temporelle"):
            current_components = frame_components[t]
            
            if not current_components:
                continue
            
            # Matrice de distance avec les événements actifs
            matched_components = set()
            
            for comp_idx, component in enumerate(current_components):
                if comp_idx in matched_components:
                    continue
                
                best_match = None
                best_score = float('inf')
                
                # Convertir le centroïde en array numpy
                current_centroid = np.array(component['centroid'])
                
                # Recherche du meilleur match avec les événements actifs
                for event_id_active, event_info in active_events.items():
                    if t - event_info['last_frame'] > 2:  # Gap temporel max
                        continue
                    
                    # Convertir le dernier centroïde en array numpy
                    last_centroid = np.array(event_info['last_centroid'])
                    
                    # Distance centroïde
                    dist = np.linalg.norm(current_centroid - last_centroid)
                    
                    # Score combiné (distance + différence d'intensité)
                    intensity_diff = abs(component['max_intensity'] - event_info['last_max_intensity'])
                    score = dist + 0.1 * intensity_diff
                    
                    if score < best_score and dist < 15:  # Seuil de distance
                        best_score = score
                        best_match = event_id_active
                
                if best_match:
                    # Mise à jour de l'événement existant
                    active_events[best_match]['components'].append((t, component))
                    active_events[best_match]['last_frame'] = t
                    active_events[best_match]['last_centroid'] = component['centroid']
                    active_events[best_match]['last_max_intensity'] = component['max_intensity']
                    matched_components.add(comp_idx)
                else:
                    # Nouvel événement
                    active_events[event_id] = {
                        'components': [(t, component)],
                        'last_frame': t,
                        'last_centroid': component['centroid'],
                        'last_max_intensity': component['max_intensity']
                    }
                    event_id += 1
            
            # Nettoyage des événements inactifs
            inactive_events = [eid for eid, info in active_events.items() 
                            if t - info['last_frame'] > 3]
            
            for eid in inactive_events:
                temporal_events.append(active_events[eid])
                del active_events[eid]
        
        # Ajout des événements restants
        for event_info in active_events.values():
            temporal_events.append(event_info)
        
        return temporal_events
    
    def _filter_and_validate_events(self, temporal_events):
        """Filtrage et validation des événements"""
        valid_events = []
        
        for event_idx, event in enumerate(tqdm(temporal_events, desc="Validation")):
            # Calcul de la taille totale
            total_size = sum(comp['area'] for _, comp in event['components'])
            
            if total_size < self.threshold_size_3d_:
                continue
            
            # Validation par corrélation si nécessaire
            if len(event['components']) > 1:
                correlations = self._compute_event_correlations(event)
                if np.mean(correlations) < self.threshold_corr_:
                    continue
            
            # Assignation des IDs
            event_id = len(valid_events) + 1
            for t, component in event['components']:
                coords = component['coords']
                for z, y in coords:
                    self.id_connected_voxel_[t, z, y] = event_id
            
            valid_events.append(event_id)
        
        self.final_id_events_ = valid_events
    
    def _compute_event_correlations(self, event):
        """Calcul optimisé des corrélations pour un événement"""
        components = event['components']
        if len(components) < 2:
            return [1.0]
        
        correlations = []
        
        # Échantillonnage pour les gros événements
        if len(components) > 10:
            step = len(components) // 10
            components = components[::step]
        
        for i in range(len(components) - 1):
            t1, comp1 = components[i]
            t2, comp2 = components[i + 1]
            
            # Extraction des profils d'intensité simplifiés
            coords1 = comp1['coords']
            coords2 = comp2['coords']
            
            if len(coords1) > 0 and len(coords2) > 0:
                # Corrélation basée sur les intensités moyennes
                int1 = self.av_[t1][coords1[:, 0], coords1[:, 1]]
                int2 = self.av_[t2][coords2[:, 0], coords2[:, 1]]
                
                corr = np.corrcoef(int1.mean(), int2.mean())[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return correlations if correlations else [0.0]
    
    def _post_process_small_regions(self):
        """Post-traitement optimisé des petites régions"""
        # Utilisation de skimage pour le nettoyage rapide
        binary_mask = self.id_connected_voxel_ > 0
        
        # Suppression des petites composantes 3D
        for t in range(self.time_length_):
            if np.any(binary_mask[t]):
                # Nettoyage 2D par frame
                cleaned = remove_small_objects(
                    binary_mask[t], 
                    min_size=self.threshold_size_3d_removed_,
                    connectivity=2
                )
                
                # Mise à jour du masque
                removed_mask = binary_mask[t] & ~cleaned
                self.id_connected_voxel_[t][removed_mask] = 0
        
        # Mise à jour de la liste des événements finaux
        unique_ids = np.unique(self.id_connected_voxel_[self.id_connected_voxel_ > 0])
        self.final_id_events_ = unique_ids.tolist()
    
    def get_results(self) -> Tuple[np.ndarray, List[int]]:
        """Retourne les résultats de la détection"""
        return self.id_connected_voxel_, self.final_id_events_
    
    def get_statistics(self) -> dict:
        """Calcule les statistiques des événements détectés"""
        stats = {
            'nb_events': len(self.final_id_events_),
            'event_sizes': [],
            'total_event_voxels': 0
        }
        
        for event_id in self.final_id_events_:
            size = np.sum(self.id_connected_voxel_ == event_id)
            stats['event_sizes'].append(size)
            stats['total_event_voxels'] += size
        
        if stats['event_sizes']:
            stats['mean_event_size'] = np.mean(stats['event_sizes'])
            stats['median_event_size'] = np.median(stats['event_sizes'])
            stats['max_event_size'] = np.max(stats['event_sizes'])
            stats['min_event_size'] = np.min(stats['event_sizes'])
        
        return stats


def detect_calcium_events_turbo(av_data: np.ndarray, 
                               params_values: dict = None,
                               save_results: bool = False,
                               output_directory: str = None,
                               use_gpu: bool = False,
                               n_jobs: int = -1) -> Tuple[np.ndarray, List[int]]:
    """
    Fonction optimisée pour la détection d'événements calciques
    """
    
    # Paramètres par défaut
    if params_values is None:
        params_values = {
            'events_extraction': {
                'threshold_size_3d': 10,
                'threshold_size_3d_removed': 5,
                'threshold_corr': 0.5
            },
            'files': {'save_results': 0},
            'paths': {'output_dir': './output'}
        }
    
    threshold_size_3d = int(params_values['events_extraction']['threshold_size_3d'])
    threshold_size_3d_removed = int(params_values['events_extraction']['threshold_size_3d_removed'])
    threshold_corr = float(params_values['events_extraction']['threshold_corr'])
    
    # Initialisation du détecteur turbo
    detector = EventDetectorTurbo(
        av_data, 
        threshold_size_3d,
        threshold_size_3d_removed, 
        threshold_corr,
        use_gpu=use_gpu,
        n_jobs=n_jobs
    )
    
    # Détection des événements
    detector.find_events()
    id_connections, id_events = detector.get_results()
    
    # Sauvegarde si demandée
    if save_results:
        if output_directory is None:
            output_directory = params_values['paths']['output_dir']
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        
    
    # Statistiques
    stats = detector.get_statistics()
    print(f"Statistiques: {stats['nb_events']} événements détectés")
    if stats['nb_events'] > 0:
        print(f"Taille moyenne: {stats['mean_event_size']:.1f} voxels")
        print(f"Taille médiane: {stats['median_event_size']:.1f} voxels")
    
    print("=" * 60)
    return id_connections, id_events


def test_with_large_synthetic_data():
    """Test avec des données synthétiques de grande taille"""
    print("=== TEST AVEC DONNÉES SYNTHÉTIQUES GRANDES ===")
    
    # Données de taille réaliste
    shape = (10, 30, 500, 300)
    print(f"Création de données de test: {shape}")
    
    av_data = np.zeros(shape, dtype=np.float32)
    
    # Ajout d'événements synthétiques
    np.random.seed(42)
    
    # Événement 1: Grand événement temporel
    t_start, t_end = 1, 2
    z_start, z_end = 5, 15
    y_start, y_end = 100, 150
    x_start, x_end = 50, 100
    
    for t in range(t_start, t_end):
        intensity = np.exp(-(t - 17)**2 / 20)  # Profil gaussien temporel
        av_data[t, z_start:z_end, y_start:y_end, x_start:x_end] = \
            intensity * (np.random.rand(z_end-z_start, y_end-y_start, x_end-x_start) * 0.3 + 0.7)
    
    # Événement 2: Événement plus petit
    t_start, t_end = 4, 5
    z_start, z_end = 20, 25
    y_start, y_end = 300, 330
    x_start, x_end = 200, 230
    
    for t in range(t_start, t_end):
        intensity = np.exp(-(t - 45)**2 / 15)
        av_data[t, z_start:z_end, y_start:y_end, x_start:x_end] = \
            intensity * (np.random.rand(z_end-z_start, y_end-y_start, x_end-x_start) * 0.4 + 0.6)
    
    # Événement 3: Événement court mais intense
    t_start, t_end = 7, 9
    z_start, z_end = 10, 18
    y_start, y_end = 400, 420
    x_start, x_end = 100, 120
    
    for t in range(t_start, t_end):
        av_data[t, z_start:z_end, y_start:y_end, x_start:x_end] = \
            np.random.rand(z_end-z_start, y_end-y_start, x_end-x_start) * 0.5 + 0.8
    
    print(f"Voxels non-nuls: {np.count_nonzero(av_data):,}")
    print(f"Plage de valeurs: [{av_data.min():.3f}, {av_data.max():.3f}]")
    
    # Test de la détection
    start_time = time.time()
    results = detect_calcium_events_turbo(
        av_data, 
        use_gpu=False,  # Changez en True si vous avez CuPy
        n_jobs=-1
    )
    total_time = time.time() - start_time
    
    print(f"Temps de traitement total: {total_time:.2f}s")
    print(f"Débit: {av_data.size / total_time / 1e6:.2f} M voxels/s")
    
    return results


if __name__ == "__main__":
    # Test avec données synthétiques grandes
    test_results = test_with_large_synthetic_data()