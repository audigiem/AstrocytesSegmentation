# AstroCa

Segmentation des astrocytes à partir d'images de fluorescence 3D+temps.

## Installation

Clone le dépôt :

```bash
git clone https://gitlab.inria.fr/anbadoua/analyzeastrocasignals.git
cd analyzeastrocasignals
```

Installe les dépendances et crée un environnement virtuel géré par Poetry :
Verifie que Poetry est installé :
```bash
poetry --version
```

Si ce n'est pas le cas, installe-le :
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

```bash
poetry install
```

## Exécution

Active l'environnement Poetry :

```bash
poetry env list
poetry env activate
```

(optionnel) Adapte les paramètres du script principal dans le fichier `config.ini` 


Exécute le script principal qui lance toute la chaîne de traitement :

```bash
chmod +x tests/main.py
poetry run ./tests/main.py
```

options disponibles pour profilage temporel:
```bash
poetry run ./tests/main.py --stats
```

option disponible pour execution sans traces
```bash
poetry run ./tests/main.py --quiet
```

options disponibles pour profilage mémoire et temporel:
```bash
poetry run ./tests/main.py --memstats
```
ATTENTION: le profilage mémoire est plus couteux en temps de calcul, prévoir un temps d'execution plus long.


## Générer la documentation

### Vérifier que doxygen est installé

```bash
doxygen --version
```
Si ce n'est pas le cas, installez-le :

```bash
sudo apt install doxygen
```

### Génération de la documentation

```bash
doxygen Doxyfile
```

### Visualiser la documentation
Ouvrez le fichier `docs/html/index.html` dans votre navigateur.
Une documentation LaTeX est également disponible dans `docs/latex/`.
Pour compiler la documentation LaTeX, utilisez :

```bash
cd docs/latex
make all
```

