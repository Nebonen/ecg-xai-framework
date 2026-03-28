"""Preprocess the ECG Arrhythmia Database and cache as .npy files.

Run once before training:
    conda activate ecg-xai
    python preprocess_dataset.py

Phase A — Build metadata CSV (data/processed/arrhythmia_metadata.csv):
    Scans all .hea files recursively, parses #Dx:, #Age:, #Sex: comments,
    maps SNOMED-CT codes to acronyms, and saves a metadata DataFrame.

Phase B — Preprocess and cache .npy files (data/preprocessed/500/):
    For each record: wfdb.rdsamp() → transpose to [12, 5000] → bandpass
    filter + z-score normalisation → save as .npy.
"""

import glob
import os
import numpy as np
import pandas as pd
import wfdb
import yaml
from tqdm import tqdm

from src.data.preprocessing import preprocess


def parse_hea_comments(hea_path: str) -> dict:
    """Extract #Dx, #Age, #Sex from a .hea file's comment lines."""
    info = {'snomed_codes': [], 'age': None, 'sex': None}
    with open(hea_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#Dx:'):
                codes = line.replace('#Dx:', '').strip()
                info['snomed_codes'] = [c.strip() for c in codes.split(',') if c.strip()]
            elif line.startswith('#Age:'):
                age_str = line.replace('#Age:', '').strip()
                try:
                    info['age'] = int(age_str)
                except ValueError:
                    info['age'] = None
            elif line.startswith('#Sex:'):
                info['sex'] = line.replace('#Sex:', '').strip()
    return info


def build_metadata(data_dir: str, snomed_mapping_file: str) -> pd.DataFrame:
    """Scan all .hea files and build a metadata DataFrame."""
    # Load SNOMED-CT code → acronym mapping
    mapping_path = os.path.join(data_dir, snomed_mapping_file)
    if os.path.exists(mapping_path):
        snomed_df = pd.read_csv(mapping_path)
        # Columns: Acronym Name, Full Name, Snomed_CT
        code_col = snomed_df.columns[2]    # Snomed_CT
        abbrev_col = snomed_df.columns[0]  # Acronym Name
        code_to_abbrev = dict(zip(
            snomed_df[code_col].astype(str),
            snomed_df[abbrev_col].astype(str),
        ))
    else:
        print(f"Warning: SNOMED mapping file not found at {mapping_path}")
        code_to_abbrev = {}

    # Discover all .hea files
    hea_pattern = os.path.join(data_dir, 'WFDBRecords', '**', '*.hea')
    hea_files = sorted(glob.glob(hea_pattern, recursive=True))

    if not hea_files:
        # Try without WFDBRecords subdirectory
        hea_pattern = os.path.join(data_dir, '**', '*.hea')
        hea_files = sorted(glob.glob(hea_pattern, recursive=True))

    print(f"Found {len(hea_files):,} .hea files")

    rows = []
    for hea_path in tqdm(hea_files, desc="Scanning .hea files"):
        record_path = hea_path[:-4]  # strip .hea extension
        record_id = os.path.basename(record_path)
        rel_path = os.path.relpath(record_path, data_dir)

        info = parse_hea_comments(hea_path)

        # Map SNOMED codes to abbreviations
        abbrevs = []
        for code in info['snomed_codes']:
            abbrev = code_to_abbrev.get(code, code)
            abbrevs.append(abbrev)

        rows.append({
            'record_id': record_id,
            'record_path': rel_path,
            'snomed_codes': ','.join(info['snomed_codes']),
            'abbreviations': ','.join(abbrevs),
            'age': info['age'],
            'sex': info['sex'],
        })

    df = pd.DataFrame(rows)
    return df


def main():
    with open('configs/data.yaml') as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg['dataset']['raw_dir']
    sampling_rate = cfg['signal']['sample_rate']
    duration = cfg['signal']['duration']
    snomed_mapping = cfg['labels']['snomed_mapping']

    # Phase A — Build metadata CSV
    print("=" * 60)
    print("Phase A: Building metadata CSV")
    print("=" * 60)

    metadata_dir = os.path.join('data', 'processed')
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, 'arrhythmia_metadata.csv')

    if os.path.exists(metadata_path):
        print(f"Metadata CSV already exists: {metadata_path}")
        meta = pd.read_csv(metadata_path)
    else:
        meta = build_metadata(data_dir, snomed_mapping)
        meta.to_csv(metadata_path, index=False)
        print(f"Saved metadata CSV: {metadata_path} ({len(meta):,} records)")

    # Print class frequency analysis
    print("\nClass frequency analysis (top 20):")
    all_abbrevs = []
    for abbrevs in meta['abbreviations'].dropna():
        all_abbrevs.extend(abbrevs.split(','))
    freq = pd.Series(all_abbrevs).value_counts()
    print(freq.head(20).to_string())

    # Phase B — Preprocess and cache .npy files
    print("\n" + "=" * 60)
    print("Phase B: Preprocessing and caching .npy files")
    print("=" * 60)

    cache_dir = os.path.join('data', 'preprocessed', str(sampling_rate))
    os.makedirs(cache_dir, exist_ok=True)

    skipped = 0
    errors = 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Preprocessing"):
        out_path = os.path.join(cache_dir, f"{row['record_id']}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            record_path = os.path.join(data_dir, row['record_path'])
            signal, fields = wfdb.rdsamp(record_path)

            # wfdb returns [timesteps, leads] — transpose to [leads, timesteps]
            signal = signal.T.astype(np.float32)

            # Preprocess each lead independently
            signal = np.stack([
                preprocess(signal[i], fs=float(sampling_rate), duration=float(duration))
                for i in range(signal.shape[0])
            ])

            np.save(out_path, signal)
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error processing {row['record_id']}: {e}")

    total = len(meta)
    processed = total - skipped - errors
    print(f"\nDone. {processed} new + {skipped} cached + {errors} errors = {total} total records")
    print(f"Cache directory: {cache_dir}/")


if __name__ == '__main__':
    main()
