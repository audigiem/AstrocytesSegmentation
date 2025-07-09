#! /usr/bin/env python3
import unittest
from astroca.activeVoxels.spaceMorphology import closing_morphology_in_space
from astroca.tools.loadData import load_data
import numpy as np

class MorphologyClosingTest(unittest.TestCase):


    def create_test_data_4d(self, shape=(2, 5, 5, 5)):
        """
        Crée une petite image 4D de test avec des motifs simples.
        @param shape: Tuple (T, Z, Y, X) définissant la taille des données
        @return: numpy array 4D avec des motifs testables
        """
        data = np.zeros(shape, dtype=np.uint8)

        # Ajout de motifs simples
        # Cube central pour t=0
        data[0, 1:-1, 1:-1, 1:-1] = 255

        # Cube décalé vers un bord pour t=1
        data[1, 1:-1, 0:-2, 0:-2] = 255

        return data

    def compare_closing_results(self, java_data, python_data, radius=1, tolerance=0):
        """
        Compare les résultats Java et Python pixel par pixel.
        @param java_data: Données traitées par l'implémentation Java (format T,Z,Y,X)
        @param python_data: Données traitées par l'implémentation Python
        @param radius: Rayon utilisé pour le traitement
        @param tolerance: Tolérance pour les différences (en pixels)
        @return: Dictionnaire avec les statistiques de comparaison
        """
        java_data = load_data(java_data)
        python_data = load_data(python_data)
        if java_data.shape != python_data.shape:
            # crop les données pour qu'elles aient la même forme
            min_shape = np.minimum(java_data.shape, python_data.shape)
            java_data = java_data[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            python_data = python_data[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
        
        differences = java_data != python_data
        diff_count = np.sum(differences)
        total_pixels = java_data.size

        # Trouver les coordonnées des pixels différents
        diff_coords = np.argwhere(differences)

        return {
            'total_pixels': total_pixels,
            'diff_count': diff_count,
            'diff_percentage': 100 * diff_count / total_pixels,
            'diff_coords': diff_coords,
            'java_shape': java_data.shape,
            'python_shape': python_data.shape
        }

    def test_closing_implementations(self):
        # 1. Créer des petites données de test
        test_data = self.create_test_data_4d(shape=(2, 5, 5, 5))  # 2 temps, 5x5x5
        radius = 1

        # 2. Appliquer l'implémentation Python (votre version)
        python_result = closing_morphology_in_space(test_data, radius, border_mode='edge')

        # 3. Simuler l'implémentation Java (vous devrez adapter cette partie)
        # Pour tester, nous allons créer un résultat Java simulé avec des différences aux bords
        java_result = np.copy(python_result)

        # Simuler des différences aux bords
        java_result[:, 0, :, :] = 0  # Bord Z min
        java_result[:, -1, :, :] = 0  # Bord Z max
        java_result[:, :, 0, :] = 0  # Bord Y min
        java_result[:, :, -1, :] = 0  # Bord Y max
        java_result[:, :, :, 0] = 0  # Bord X min
        java_result[:, :, :, -1] = 0  # Bord X max

        # 4. Comparer les résultats
        comparison = self.compare_closing_results(java_result, python_result, radius)

        # 5. Afficher les résultats
        print("\n=== Résultats de la comparaison ===")
        print(f"Dimensions Java: {comparison['java_shape']}")
        print(f"Dimensions Python: {comparison['python_shape']}")
        print(
            f"Pixels différents: {comparison['diff_count']}/{comparison['total_pixels']} ({comparison['diff_percentage']:.2f}%)")
        print("Coordonnées des différences (T,Z,Y,X):")
        print(comparison['diff_coords'])

        return comparison
    
if __name__ == '__main__':
    # unittest.main()
    # # Pour exécuter ce test, utilisez la commande suivante dans le terminal:
    # # python -m unittest tests/componentTest/morphologyClosing.py
    # # Assurez-vous que le chemin vers le fichier est correct.
    expected_data = "/home/matteo/Bureau/INRIA/codeJava/outputdir20/Closing_in_space.tif"
    python_data = "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/filledSpaceMorphology.tif"
    result = MorphologyClosingTest().compare_closing_results(expected_data, python_data)
    print("\n=== Résultats de la comparaison ===")
    print(f"Dimensions Java: {result['java_shape']}")
    print(f"Dimensions Python: {result['python_shape']}")
    print(f"Pixels différents: {result['diff_count']}/{result['total_pixels']} ({result['diff_percentage']:.2f}%)")
    print("Coordonnées des différences (T,Z,Y,X):")
    print(result['diff_coords'])    
    


    
    