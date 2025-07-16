def compare_files(file1_path, file2_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print(f"Les fichiers ont un nombre de lignes différent : {len(lines1)} vs {len(lines2)}")
        # tronquer les lignes pour comparaison
        min_length = min(len(lines1), len(lines2))
        lines1 = lines1[:min_length]
        lines2 = lines2[:min_length]

    all_match = True
    for i, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
        parts1 = line1.strip().split()
        parts2 = line2.strip().split()

        if len(parts1) != 3 or len(parts2) != 3:
            print(f"Ligne {i} mal formée :\n  {line1}\n  {line2}")
            all_match = False
            continue

        cat1, id1, val1 = parts1
        cat2, id2, val2 = parts2

        if (cat1 != cat2) or (id1 != id2) or (val1 != val2):
            print(f"Différence à la ligne {i} :")
            print(f"  Cible Java : {cat1} {id1} {val1}")
            print(f"  Mon résultat : {cat2} {id2} {val2}")
            all_match = False

    if all_match:
        print("Les fichiers sont identiques ligne par ligne.")
    return all_match


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        file2 = "/home/matteo/Bureau/INRIA/codePython/AstrocytesSegmentation/sizeEventPython.txt"
        file1 = "/home/matteo/Bureau/INRIA/codePython/AstrocytesSegmentation/SizeEventJava.txt"
        compare_files(file1, file2) 
    
    
    elif len(sys.argv) != 3:
        print("Usage: python comparetextFile.py <file1> <file2>")
        sys.exit(1)
        
    else:
        
        file1 = sys.argv[1]
        file2 = sys.argv[2]

        compare_files(file1, file2)