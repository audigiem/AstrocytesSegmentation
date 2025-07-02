import numpy as np
from scipy import ndimage
from scipy.signal import correlate
from sklearn.neighbors import NearestNeighbors
from collections import deque
import time
import numba
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')



@jit(nopython=True, parallel=True)
def correlation_zero_boundary(v1, v2):
    """
    Compute cross-correlation with zero boundary conditions (like the Java version).
    Optimized with Numba for speed.
    """
    nV1 = len(v1)
    nV2 = len(v2)
    size = nV1 + nV2 - 1
    vout = np.zeros(size, dtype=np.float32)
    
    for n in prange(-nV2 + 1, nV1):  # prange for parallelization
        sum_val = 0.0
        for m in range(nV1):
            index = m - n
            if 0 <= index < nV2:
                sum_val += v2[index] * v1[m]
        vout[n + nV2 - 1] = sum_val
    
    return vout

@jit(nopython=True)
def compute_normalized_cross_correlation(v1, v2):
    """
    Compute normalized cross-correlation (NCC) for all lags, equivalent to the Java version.
    Returns an array of NCC values for each possible lag.
    Optimized with Numba.
    """
    if len(v1) == 0 or len(v2) == 0:
        return np.zeros(1, dtype=np.float32)  # Return empty array
    
    # Cross-correlation
    vout = correlation_zero_boundary(v1, v2)
    
    # Auto-correlation of v1 at lag 0
    auto_v1 = np.sum(v1 ** 2)  # Equivalent to correlation_zero_boundary(v1,v1)[len(v1)-1]
    
    # Auto-correlation of v2 at lag 0
    auto_v2 = np.sum(v2 ** 2)  # Equivalent to correlation_zero_boundary(v2,v2)[len(v2)-1]
    
    # Normalization factor
    den = np.sqrt(auto_v1 * auto_v2)
    
    # Avoid division by zero
    if den == 0:
        return np.zeros_like(vout)
    
    # Normalize all correlation values
    vout /= den
    
    return vout
    


class VoxelGroupingAlgorithm:
    """
    Implémentation optimisée de l'algorithme de regroupement de voxels actifs
    basé sur la corrélation croisée normalisée et la croissance de région.
    """
    
    def __init__(self, params: dict):
        """
        Paramètres:
        - correlation_threshold: seuil de corrélation pour grouper les voxels (0.6-0.7)
        - min_group_size: taille minimale d'un groupe principal (400 voxels)
        - small_group_threshold: seuil pour identifier les petits groupes (40 voxels)
        - tiny_group_threshold: seuil pour supprimer les groupes trop petits (20 voxels)
        """
        required_keys = {'events_extraction'}
        if not required_keys.issubset(params.keys()):
            raise ValueError(f"Les paramètres doivent contenir les clés: {required_keys}")
        self.correlation_threshold = float(params['events_extraction'].get('threshold_corr', 0.6))
        self.min_group_size = int(params['events_extraction'].get('threshold_size_3d', 400))
        self.small_group_threshold = int(params['events_extraction'].get('threshold_size_3d_removed', 20))
        self.tiny_group_threshold = 5
        
        # Structure pour les 26 voisins en 3D
        self.neighbors_3d = self._generate_3d_neighbors()
    
    def _generate_3d_neighbors(self):
        """Génère les 26 voisins 3D d'un voxel"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))
        return np.array(neighbors)
    
    
    def _extract_full_temporal_pattern(self, data, voxel_coords, reference_frame):
        """
        Extrait le pattern d'intensité centré sur la reference_frame, s'arrêtant quand le voxel devient inactif (0)
        dans les deux directions temporelles.
        
        Args:
            data: ndarray de forme (T, Z, Y, X) - données d'intensité
            voxel_coords: tuple (z, y, x) - coordonnées du voxel
            reference_frame: int - frame centrale de référence
            
        Returns:
            ndarray: profil temporel centré, ou None si le voxel est inactif à la frame de référence
        """
        z, y, x = voxel_coords
        t_max = data.shape[0]
        # print(data[:, z, y, x])
        # Vérifier que le voxel est actif à la frame de référence
        if data[reference_frame, z, y, x] == 0:
            return []
        
        # Trouver le début actif (en remontant dans le temps)
        start_frame = reference_frame
        while start_frame >= 0 and data[start_frame, z, y, x] != 0:
            start_frame -= 1
        start_frame = max(0, start_frame + 1)  # Ajuster pour inclure le dernier frame actif
        
        # Trouver la fin active (en avançant dans le temps)
        end_frame = reference_frame
        while end_frame < t_max and data[end_frame, z, y, x] != 0:
            end_frame += 1
        end_frame = min(t_max - 1, end_frame - 1)  # Ajuster pour inclure le dernier frame actif
        
        # Extraire le segment temporel
        pattern = data[start_frame:end_frame + 1, z, y, x]
        
        return pattern, start_frame, end_frame
    
   
    
    def _get_valid_neighbors_3d(self, coords, shape_range):
        """
        Retourne les coordonnées valides des voisins 3D d'un voxel
        """
        t, z, y, x = coords
        z_max, y_max, x_max = shape_range
        
        valid_neighbors = []
        
        for dx, dy, dz in self.neighbors_3d:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < x_max and 0 <= ny < y_max and 0 <= nz < z_max):
                valid_neighbors.append((t, nz, ny, nx))
        
        return valid_neighbors
    
    def _find_seed_point_on_frame(self, data, active_voxels_mask, assigned_voxels, frame):
        """
        Trouve le seed point avec l'intensité MAXIMALE parmi les voxels actifs non assignés
        sur une frame donnée, en utilisant des opérations vectorisées NumPy.
        
        Args:
            data: ndarray de forme (T, Z, Y, X) contenant les intensités
            active_voxels_mask: ndarray booléen de même forme que data
            assigned_voxels: set des coordonnées spatiales (z,y,x) déjà assignées
            frame: indice de la frame à analyser
        
        Returns:
            tuple: (z, y, x) du voxel avec intensité maximale, ou None si aucun trouvé
        """
        # 1. Obtenir les indices des voxels actifs sur cette frame
        active_indices = np.where(active_voxels_mask[frame])
        
        if len(active_indices[0]) == 0:
            return None
        
        # 2. Créer un masque des voxels non assignés (coordonnées spatiales)
        coords_list = list(zip(active_indices[0], active_indices[1], active_indices[2]))
        unassigned_mask = np.array([coord not in assigned_voxels for coord in coords_list])
        
        if not np.any(unassigned_mask):
            return None
        
        # 3. Extraire les intensités des voxels non assignés
        intensities = data[frame, active_indices[0][unassigned_mask], 
                            active_indices[1][unassigned_mask], 
                            active_indices[2][unassigned_mask]]
        
        # 4. Trouver l'indice du maximum
        max_idx = np.argmax(intensities)
        
        # 5. Récupérer les coordonnées correspondantes
        seed_coords = (active_indices[0][unassigned_mask][max_idx],
                    active_indices[1][unassigned_mask][max_idx],
                    active_indices[2][unassigned_mask][max_idx])
        
        return seed_coords
    
    def _region_growing(self, data, active_voxels_mask, assigned_voxels, seed_coords, reference_frame):
        """
        Algorithme de croissance de région basé sur la corrélation.
        Gère maintenant les coordonnées spatio-temporelles.
        """
        group = set()
        to_investigate = deque()
        investigated = set()
        
        # Pattern de référence du seed (signal temporel complet)
        seed_pattern, start_frame, end_frame = self._extract_full_temporal_pattern(data, seed_coords, reference_frame)
        
        if len(seed_pattern) == 0:
            return group
        
        if start_frame is None and end_frame is None:
            return group
            
        z, y, x = seed_coords
        
        # Ajouter toutes les coordonnées spatio-temporelles du seed
        for t in range(start_frame, end_frame + 1):
            spatio_temporal_coord = (t, z, y, x)
            group.add(spatio_temporal_coord)
            to_investigate.append((t, z, y, x))
            
        # Marquer les coordonnées spatiales comme traitées pour éviter les doublons
        spatial_investigated = set()
        spatial_investigated.add((t, z, y, x))
        
        while to_investigate:
            # Extraire les coordonnées spatio-temporelles du voxel à investiguer
            # (t, z, y, x) = to_investigate.popleft()
            current_coords = to_investigate.popleft()
            
            if current_coords in investigated:
                continue
            investigated.add(current_coords)

            # Obtenir les voisins valides en 3D spatial
            neighbors = self._get_valid_neighbors_3d(current_coords, data.shape[1:4])
            
            for neighbor_coords in neighbors:
                # Vérifier si le voisin spatial n'est pas déjà assigné et pas déjà dans le groupe
                if (neighbor_coords not in assigned_voxels and 
                    neighbor_coords not in spatial_investigated):
                    
                    # Vérifier si ce voxel est actif à un moment donné
                    nt, nz, ny, nx = neighbor_coords
                    is_active = active_voxels_mask[nt, nz, ny, nx] > 0
                    if not is_active:
                        continue
                    if is_active:
                        # Extraire le pattern temporel complet du voisin
                        neighbor_pattern, neighbor_start, neighbor_end = self._extract_full_temporal_pattern(data, voxel_coords=(nz, ny, nx), reference_frame=nt)
                        
                        if len(neighbor_pattern) == 0:
                            continue

                        if start_frame is None and end_frame is None:
                            continue

                        # Calculer la corrélation
                        correlation = compute_normalized_cross_correlation(
                            seed_pattern, neighbor_pattern
                        )
                        
                        # Si la corrélation dépasse le seuil, ajouter TOUTES ses coordonnées temporelles
                        if np.max(correlation) >= self.correlation_threshold:                            
                            # Ajouter toutes les coordonnées spatio-temporelles du voisin
                            for t in range(neighbor_start, neighbor_end + 1):
                                spatio_temporal_coord = (t, nz, ny, nx)
                                group.add(spatio_temporal_coord)

                            # Ajouter à la queue d'investigation et marquer comme traité
                            to_investigate.append(neighbor_coords)
                            spatial_investigated.add(neighbor_coords)
        
        return group
    
    def _merge_small_groups_spatio_temporal(self, groups, data):
        """
        Fusionne les petits groupes avec leurs voisins les plus proches.
        Adaptée pour les coordonnées spatio-temporelles.
        """
        if not groups:
            return groups
            
        # Identifier les petits groupes et les grands groupes
        small_groups = []
        large_groups = []
        
        for group in groups:
            if len(group) < self.min_group_size:
                small_groups.append(group)
            else:
                large_groups.append(group)
        
        if not small_groups:
            return groups
        
        # Commencer avec les grands groupes
        final_groups = large_groups.copy()
        
        # Traiter chaque petit groupe
        for small_group in small_groups:
            if len(small_group) < self.tiny_group_threshold:
                # Supprimer les groupes trop petits
                continue
            
            # Trouver le groupe le plus proche spatialement
            best_match_idx = None
            best_distance = float('inf')
            
            if len(small_group) == 0:
                continue
            
            # Calculer le centroïde spatial du petit groupe (ignorer la dimension temporelle)
            spatial_coords = [(z, y, x) for (t, z, y, x) in small_group]
            if not spatial_coords:
                continue
                
            small_centroid = np.mean(spatial_coords, axis=0)
            
            # Chercher parmi tous les groupes finaux existants
            for idx, large_group in enumerate(final_groups):
                if len(large_group) == 0:
                    continue
                
                # Calculer le centroïde spatial du grand groupe
                large_spatial_coords = [(z, y, x) for (t, z, y, x) in large_group]
                if not large_spatial_coords:
                    continue
                    
                large_centroid = np.mean(large_spatial_coords, axis=0)
                
                # Distance euclidienne
                distance = np.linalg.norm(small_centroid - large_centroid)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_idx = idx
            
            # Fusionner avec le groupe le plus proche ou garder séparément
            if (best_match_idx is not None and 
                best_distance < 50):  # Seuil de distance spatiale
                
                final_groups[best_match_idx] = final_groups[best_match_idx].union(small_group)
            elif len(small_group) >= self.small_group_threshold:
                # Garder comme groupe séparé si assez grand
                final_groups.append(small_group)
        
        return final_groups
    
    def group_voxels(self, data, active_voxels_mask):
        """
        Fonction principale pour grouper les voxels actifs.
        Gère maintenant les coordonnées spatio-temporelles.
        
        Paramètres:
        - data: np.ndarray de forme (T, z, y, x) contenant les données d'intensité
        - active_voxels_mask: np.ndarray booléen de même forme indiquant les voxels actifs
        
        Retourne:
        - groups: liste de sets contenant les coordonnées spatio-temporelles (t,x,y,z) de chaque groupe
        - group_labels: np.ndarray avec les labels des groupes pour chaque voxel
        """
        print("Démarrage du regroupement de voxels...")
        start_time = time.time()
        print(f"Forme des données: {data.shape}")
        
        groups = []
        assigned_voxels = set()  # Stocke les coordonnées spatiales (x,y,z) déjà assignées
        
        # Traitement frame par frame dans l'ordre chronologique
        for frame in range(data.shape[0]):
            print(f"Traitement de la frame {frame+1}/{data.shape[0]}")
            
            # Continuer à chercher des seed points sur cette frame
            while True:
                # Trouver le seed point avec l'intensité maximale sur cette frame
                seed_coords = self._find_seed_point_on_frame(
                    data, active_voxels_mask, assigned_voxels, frame
                )
                
                if seed_coords is None:
                    # Plus de seed points disponibles sur cette frame
                    break
                
                print(f"  Seed trouvé: {seed_coords}, t={frame}, id={len(groups) + 1}")
                
                # Croissance de région à partir de ce seed
                group = self._region_growing(
                    data, active_voxels_mask, assigned_voxels, seed_coords, frame
                )
                
                if len(group) > 0:
                    groups.append(group)
                    # Extraire les coordonnées spatiales uniques du groupe pour assigned_voxels
                    spatial_coords = set()
                    for t, z, y, x in group:
                        spatial_coords.add((z, y, x))
                    assigned_voxels.update(spatial_coords)
                    print(f"  Nouveau groupe trouvé: {len(group)} voxels spatio-temporels, {len(spatial_coords)} voxels spatiaux")
                else:
                    # Marquer ce voxel comme assigné même s'il ne forme pas de groupe
                    assigned_voxels.add(seed_coords)
        
        print(f"Nombre de groupes avant fusion: {len(groups)}")
        
        # Fusionner les petits groupes
        final_groups = self._merge_small_groups_spatio_temporal(groups, data)
        
        print(f"Nombre de groupes après fusion: {len(final_groups)}")
        
        # Créer les labels spatio-temporels
        group_labels = np.zeros(data.shape, dtype=np.int32)
        
        for group_id, group in enumerate(final_groups, 1):
            for t, z, y, x in group:
                group_labels[t, z, y, x] = group_id
        
        # Statistiques finales
        group_sizes = [len(group) for group in final_groups]
        if group_sizes:
            print(f"Tailles des groupes: {group_sizes}")

        end_time = time.time()
        print(f"Temps d'exécution: {end_time - start_time:.2f} secondes")

        return final_groups, group_labels

# Exemple d'utilisation
def example_usage():
    """
    Exemple d'utilisation de l'algorithme
    """
    # Créer des données d'exemple (remplacez par vos vraies données)
    print("Création de données d'exemple...")
    
    # Pour l'exemple, créons des données plus petites
    T, z, y, x = 10, 20, 50, 30
    data = np.random.randn(T, z, y, x) * 100 + 500
    
    # Créer des voxels actifs d'exemple
    active_voxels_mask = np.random.rand(T, z, y, x) > 0.95
    
    params = {
        'events_extraction': {
            'threshold_size_3d': 10,  # Réduit pour l'exemple
            'threshold_size_3d_removed': 5,  # Réduit pour l'exemple
            'threshold_corr': 0.3  # Réduit pour l'exemple
        }
    }
    
    # Initialiser l'algorithme
    algorithm = VoxelGroupingAlgorithm(params)
    
    # Exécuter l'algorithme
    groups, group_labels = algorithm.group_voxels(data, active_voxels_mask)
    
    print(f"Regroupement terminé avec {len(groups)} groupes")
    print(f"Forme des labels: {group_labels.shape}")
    
    return groups, group_labels

if __name__ == "__main__":
    groups, labels = example_usage()