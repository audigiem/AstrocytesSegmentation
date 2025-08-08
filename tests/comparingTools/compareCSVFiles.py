#!/usr/bin/env python3
"""
@file compareCSVFiles.py
@brief Compare two CSV files and print the differences, assuming same column order.
"""

import sys
import pandas as pd
import numpy as np
import csv


def detect_separator(file_path: str) -> str:
    with open(file_path, newline="") as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter


def normalize_dataframe(df: pd.DataFrame, float_precision: int = 5) -> pd.DataFrame:
    # Convert all numeric columns to float64 and round them
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(np.float64).round(float_precision)
        else:
            df[col] = df[col].astype(str)
    return df


def compare_csv_files(file1: str, file2: str, float_precision: int = 5) -> None:
    sep1 = detect_separator(file1)
    sep2 = detect_separator(file2)

    df1 = pd.read_csv(file1, sep=sep1)
    df2 = pd.read_csv(file2, sep=sep2)

    if df1.shape != df2.shape:
        print(f"Shape mismatch: file1={df1.shape}, file2={df2.shape}")
        return

    df1 = normalize_dataframe(df1, float_precision)
    df2 = normalize_dataframe(df2, float_precision)

    diffs = []
    for i, (row1, row2) in enumerate(
        zip(df1.itertuples(index=False), df2.itertuples(index=False))
    ):
        if tuple(row1) != tuple(row2):
            diffs.append((i, tuple(row1), tuple(row2)))

    if not diffs:
        print("The two CSV files are identical (after normalization).")
    else:
        print(f"Found {len(diffs)} differing rows:")
        for i, r1, r2 in diffs:
            print(f"Line {i + 1} differs:")
            # show the columns that differ
            for col, val1, val2 in zip(df1.columns, r1, r2):
                if val1 != val2:
                    print(f"  Column '{col}': Target file: {val1}, Output file: {val2}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        expected_CSV = "/home/matteo/Bureau/INRIA/codeJava/outputdir20/Features.csv"
        actual_CSV = (
            "/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/Features.csv"
        )
        compare_csv_files(expected_CSV, actual_CSV)

    elif len(sys.argv) != 3:
        print("Usage: python compareCSVFiles.py <file1.csv> <file2.csv>")
        sys.exit(1)
    else:
        file1 = sys.argv[1]
        file2 = sys.argv[2]

        compare_csv_files(file1, file2)
        print("=" * 60)
