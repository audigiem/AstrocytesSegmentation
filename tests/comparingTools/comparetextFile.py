import numpy as np


def lire_fichier(fichier):
    with open(fichier, "r") as f:
        lignes = f.readlines()
    donnees = [
        list(map(np.float32, ligne.strip().split()))
        for ligne in lignes
        if ligne.strip()
    ]
    return donnees


def somme_par_colonne(donnees):
    # Transpose puis somme par colonne
    return [sum(col) for col in zip(*donnees)]


def comparer_fichiers(fichier1, fichier2, epsilon=1e-6):
    # Lire les fichiers
    data1 = lire_fichier(fichier1)
    data2 = lire_fichier(fichier2)

    # Vérification du nombre de lignes
    if len(data1) != len(data2):
        print(
            f"Les fichiers n'ont pas le même nombre de lignes : {len(data1)} vs {len(data2)}"
        )
        return

    # Calcul des sommes par colonne
    somme1 = somme_par_colonne(data1)
    somme2 = somme_par_colonne(data2)

    print(f"Sommes fichier 1 : {somme1}")
    print(f"Sommes fichier 2 : {somme2}")

    # Comparaison
    egales = all(abs(a - b) < epsilon for a, b in zip(somme1, somme2))
    if egales:
        print("✅ Les sommes par colonne sont égales (à epsilon près).")
    else:
        print("❌ Les sommes par colonne diffèrent.")


# Exemple d'utilisation
comparer_fichiers(
    "/home/matteo/Bureau/INRIA/codeJava/AnalyzeAstroCaSignals/output.txt",
    "/home/matteo/Bureau/INRIA/codePython/AstrocytesSegmentation/output.txt",
)
