import numpy as np
from scipy import ndimage
from scipy.signal import correlate
from sklearn.neighbors import NearestNeighbors
from collections import deque
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
            data: ndarray de forme (T, X, Y, Z) - données d'intensité
            voxel_coords: tuple (x, y, z) - coordonnées du voxel
            reference_frame: int - frame centrale de référence
            
        Returns:
            ndarray: profil temporel centré, ou None si le voxel est inactif à la frame de référence
        """
        x, y, z = voxel_coords
        t_max = data.shape[0]
        
        # Vérifier que le voxel est actif à la frame de référence
        if data[reference_frame, x, y, z] == 0:
            return []
        
        # Trouver le début actif (en remontant dans le temps)
        start_frame = reference_frame
        while start_frame >= 0 and data[start_frame, x, y, z] != 0:
            start_frame -= 1
        start_frame = max(0, start_frame + 1)  # Ajuster pour inclure le dernier frame actif
        
        # Trouver la fin active (en avançant dans le temps)
        end_frame = reference_frame
        while end_frame < t_max and data[end_frame, x, y, z] != 0:
            end_frame += 1
        end_frame = min(t_max - 1, end_frame - 1)  # Ajuster pour inclure le dernier frame actif
        
        # Extraire le segment temporel
        pattern = data[start_frame:end_frame + 1, x, y, z].copy()
        
        return pattern
    
    def _get_valid_neighbors_3d(self, coords, spatial_shape):
        """
        Retourne les coordonnées valides des voisins 3D d'un voxel
        """
        x, y, z = coords
        x_max, y_max, z_max = spatial_shape
        
        valid_neighbors = []
        
        for dx, dy, dz in self.neighbors_3d:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            if 0 <= nx < x_max and 0 <= ny < y_max and 0 <= nz < z_max:
                valid_neighbors.append((nx, ny, nz))
        
        return valid_neighbors
    
    def _find_seed_point_on_frame(self, data, active_voxels_mask, assigned_voxels, frame):
        """
        Trouve le seed point avec l'intensité MAXIMALE parmi les voxels actifs non assignés
        sur une frame donnée, en utilisant des opérations vectorisées NumPy.
        
        Args:
            data: ndarray de forme (T, X, Y, Z) contenant les intensités
            active_voxels_mask: ndarray booléen de même forme que data
            assigned_voxels: set des coordonnées déjà assignées
            frame: indice de la frame à analyser
        
        Returns:
            tuple: (x, y, z) du voxel avec intensité maximale, ou None si aucun trouvé
        """
        # 1. Obtenir les indices des voxels actifs sur cette frame
        active_indices = np.where(active_voxels_mask[frame])
        
        if len(active_indices[0]) == 0:
            return None
        
        # 2. Créer un masque des voxels non assignés
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
        Algorithme de croissance de région basé sur la corrélation
        """
        group = set()
        to_investigate = deque([seed_coords])
        investigated = set()
        
        # Pattern de référence du seed (signal temporel complet)
        seed_pattern = self._extract_full_temporal_pattern(data, seed_coords, reference_frame)
       
        
        if len(seed_pattern) == 0:
            return group
        
        group.add(seed_coords)
        
        
        while to_investigate:
            current_coords = to_investigate.popleft()
            
            if current_coords in investigated:
                continue
                
            investigated.add(current_coords)
            
            # Obtenir les voisins valides en 3D
            neighbors = self._get_valid_neighbors_3d(current_coords, data.shape[1:])
            
            for neighbor_coords in neighbors:
                # Vérifier si le voisin est un voxel actif quelque part dans le temps
                # et qu'il n'est pas déjà assigné
                if (neighbor_coords not in assigned_voxels and 
                    neighbor_coords not in group and 
                    neighbor_coords not in investigated):
                    
                    # Vérifier si ce voxel est actif à un moment donné
                    x, y, z = neighbor_coords
                    is_active = np.any(active_voxels_mask[:, x, y, z])
                    
                    if is_active:
                        # Extraire le pattern temporel complet du voisin
                        neighbor_pattern = self._extract_full_temporal_pattern(data, neighbor_coords, reference_frame)
                        
                        if len(neighbor_pattern) == 0:
                            continue
                        
                        # Calculer la corrélation
                        correlation = compute_normalized_cross_correlation(
                            seed_pattern, neighbor_pattern
                        )
                        
                        # Si la corrélation dépasse le seuil, ajouter au groupe
                        if np.max(correlation) >= self.correlation_threshold:
                            group.add(neighbor_coords)
                            to_investigate.append(neighbor_coords)
        
        return group
    
    def _merge_small_groups(self, groups, data):
        """
        Fusionne les petits groupes avec leurs voisins les plus proches
        ou les supprime s'ils sont trop petits
        """
        if not groups:
            return groups
            
        # Identifier les petits groupes et les grands groupes
        small_groups = []
        large_groups = []
        
        for i, group in enumerate(groups):
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
                
            # Calculer le centroïde du petit groupe
            small_centroid = np.mean(list(small_group), axis=0)
            
            # Chercher parmi tous les groupes finaux existants
            for idx, large_group in enumerate(final_groups):
                if len(large_group) == 0:
                    continue
                    
                # Calculer le centroïde du grand groupe
                large_centroid = np.mean(list(large_group), axis=0)
                
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
        Fonction principale pour grouper les voxels actifs
        
        Paramètres:
        - data: np.ndarray de forme (T, X, Y, Z) contenant les données d'intensité
        - active_voxels_mask: np.ndarray booléen de même forme indiquant les voxels actifs
        
        Retourne:
        - groups: liste de sets contenant les coordonnées des voxels de chaque groupe
        - group_labels: np.ndarray avec les labels des groupes pour chaque voxel
        """
        print("Démarrage du regroupement de voxels...")
        print(f"Forme des données: {data.shape}")
        
        groups = []
        assigned_voxels = set()
        
        # Traitement frame par frame dans l'ordre chronologique
        for frame in range(data.shape[0]):
            print(f"Traitement de la frame {frame+1}/{data.shape[0]}")
            
            # Continuer à chercher des seed points sur cette frame
            # jusqu'à ce qu'il n'y en ait plus
            while True:
                # Trouver le seed point avec l'intensité maximale sur cette frame
                seed_coords = self._find_seed_point_on_frame(
                    data, active_voxels_mask, assigned_voxels, frame
                )
                
                print(f"  Seed trouvé: {seed_coords}, t={frame}, id={len(groups) + 1}")
                
                if seed_coords is None:
                    # Plus de seed points disponibles sur cette frame
                    break
                
                # Croissance de région à partir de ce seed
                group = self._region_growing(
                    data, active_voxels_mask, assigned_voxels, seed_coords, frame
                )
                
                if len(group) > 0:
                    groups.append(group)
                    assigned_voxels.update(group)
                    print(f"  Nouveau groupe trouvé: {len(group)} voxels")
                else:
                    # Marquer ce voxel comme assigné même s'il ne forme pas de groupe
                    assigned_voxels.add(seed_coords)
        
        print(f"Nombre de groupes avant fusion: {len(groups)}")
        
        # Fusionner les petits groupes
        final_groups = self._merge_small_groups(groups, data)
        
        print(f"Nombre de groupes après fusion: {len(final_groups)}")
        
        # Créer les labels
        group_labels = np.zeros(data.shape[1:], dtype=np.int32)
        
        for group_id, group in enumerate(final_groups, 1):
            for x, y, z in group:
                group_labels[x, y, z] = group_id
        
        # Statistiques finales
        group_sizes = [len(group) for group in final_groups]
        if group_sizes:
            print(f"Tailles des groupes: {group_sizes}")
            print(f"Groupe le plus grand: {max(group_sizes)} voxels")
            print(f"Groupe le plus petit: {min(group_sizes)} voxels")
            print(f"Taille moyenne: {np.mean(group_sizes):.1f} voxels")
        
        return final_groups, group_labels

# Exemple d'utilisation
def example_usage():
    """
    Exemple d'utilisation de l'algorithme
    """
    # Créer des données d'exemple (remplacez par vos vraies données)
    # Forme: (T=100, X=30, Y=500, Z=300)
    print("Création de données d'exemple...")
    
    # Pour l'exemple, créons des données plus petites
    T, X, Y, Z = 10, 20, 50, 30
    data = np.random.randn(T, X, Y, Z) * 100 + 500
    
    # Créer des voxels actifs d'exemple
    active_voxels_mask = np.random.rand(T, X, Y, Z) > 0.95
    
    params = {
        'events_extraction': {
            'threshold_size_3d': 400,
            'threshold_size_3d_removed': 20,
            'threshold_corr': 0.6
        }
    }
    
    # Initialiser l'algorithme
    algorithm = VoxelGroupingAlgorithm(params)
    
    # Exécuter l'algorithme
    groups, group_labels = algorithm.group_voxels(data, active_voxels_mask)
    
    print(f"Regroupement terminé avec {len(groups)} groupes")
    
    return groups, group_labels

if __name__ == "__main__":
    groups, labels = example_usage()