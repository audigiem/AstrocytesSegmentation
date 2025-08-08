#!/bin/bash

# 1. Générer l'arborescence complète dans un fichier tmp
TMP_TREE=$(mktemp)
tree -L 2 -f --noreport > "$TMP_TREE"

# 2. Appeler un script Python pour le filtrage
python3 - <<EOF
import os
import subprocess
from pathlib import Path

def is_ignored(path):
    """Vérifie si un fichier/dossier est ignoré par git"""
    try:
        cmd = ["git", "check-ignore", "-q", str(path)]
        return subprocess.call(cmd) == 0
    except:
        return False

# Lire l'arborescence
with open("$TMP_TREE", "r") as f:
    lines = f.readlines()

# Filtrer ligne par ligne
for line in lines:
    path = line.strip().split()[-1]  # Extrait le chemin
    if not is_ignored(path):
        print(line, end="")
EOF

# 3. Nettoyer
rm "$TMP_TREE"