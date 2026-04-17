# IMPPATez
IMPPATez is a Gradio-based app for searching IMPPAT phytochemicals by plant, evaluating Lipinski Rule of Five (RO5), and downloading available 3D SDF files as a ZIP.

🔎 IMPHY ID Extractor | 📦 Automated 3D SDF Downloader | ☁️ Cloud Ready
🧬 Natural Product Informatics Tool | ⚡ IMPPAT Database Automation

## Features
- Search IMPPAT phytochemicals for a plant name
- Extract `IMPHY_ID`, phytochemical name, and SMILES
- Evaluate RO5 status using RDKit
- Export results to CSV
- Download available 3D SDF files and generate a missing-ID report
- Cache-first SDF retrieval: reuses previously downloaded `IMPHY_ID.sdf` files from earlier runs
- Automatic retry on transient SDF download failures before marking IDs as missing

## SDF download reliability behavior
When downloading 3D SDF files, the app now uses a two-step reliability flow:
1. **Cache check first**: looks for an existing valid SDF file in prior `outputs/sdf_files_*` folders and reuses it.
2. **Network retry fallback**: if no cache is found, retries IMPPAT download requests up to 3 times with short incremental backoff.

This reduces false missing-ID reports caused by temporary network/server issues.

## Requirements
- Python 3.10+
- pip

## Setup
```bash
pip install gradio requests pandas rdkit
```

## Run the app
From this directory, run:
```bash
python imppatez.py
```
The app will print a local Gradio URL in the terminal.

## Run tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Output files
Generated artifacts are saved under:
- `outputs/` (CSV exports, SDF folders, ZIP files, missing ID reports)


IMPPATez · Python · Gradio · Requests · Pandas · RDKit
Automated phytochemical extraction, RO5 check & 3D molecular structure retrieval
Nabiul Orko | Planteran 🍀
