"""Download the ECG Arrhythmia Database v1.0.0 from PhysioNet.

URL: https://physionet.org/content/ecg-arrhythmia/1.0.0/

This script uses wget to download the full dataset (~2.3 GB compressed).
wget must be installed on your system.
"""

import os
import subprocess
import sys

DEST = os.path.join(os.path.dirname(__file__), "data", "raw", "ecg-arrhythmia")
URL  = "https://physionet.org/files/ecg-arrhythmia/1.0.0/"

os.makedirs(DEST, exist_ok=True)

# Check wget is available
if subprocess.call(["which", "wget"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
    print("wget not found.  Install it with:  brew install wget")
    sys.exit(1)

cmd = [
    "wget",
    "--recursive",          # download the full directory tree
    "--no-clobber",         # skip files already downloaded
    "--continue",           # resume partial downloads
    "--no-parent",          # don't follow links above the URL
    "--no-host-directories",# don't create physionet.org/ subfolder
    "--cut-dirs=3",         # strip /files/ecg-arrhythmia/1.0.0/ prefix
    f"--directory-prefix={DEST}",
    URL,
]

print(f"\nDownloading ECG Arrhythmia Database (~2.3 GB) to {DEST}/")
subprocess.run(cmd, check=True)
print(f"\nDone.  Files are in {DEST}/")
