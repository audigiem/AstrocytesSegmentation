# AstroCa

Segmentation des astrocytes à partir d'images de fluorescence 3D+temps.

## Installation

Clone le dépôt :

```bash
git clone https://gitlab.inria.fr/anbadoua/analyzeastrocasignals.git
cd analyzeastrocasignals
```

Crée un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Exécution

Exécute le script principal :

```bash
chmod +x tests/main.py
./tests/main.py
```

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

