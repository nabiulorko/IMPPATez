# IMPPATez
IMPPATez is a Gradio-based app for searching IMPPAT phytochemicals by plant, evaluating Lipinski Rule of Five (RO5), and downloading available 3D SDF files as a ZIP.

## Features
- Search IMPPAT phytochemicals for a plant name
- Extract `IMPHY_ID`, phytochemical name, and SMILES
- Evaluate RO5 status using RDKit
- Export results to CSV
- Download available 3D SDF files and generate a missing-ID report

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

