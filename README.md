# 🌿 IMPPATez — Natural Product Docking Pipeline

**IMPPATez** is a Streamlit-based, end-to-end virtual screening pipeline for docking phytochemicals from the **IMPPAT** (Indian Medicinal Plants, Phytochemistry And Therapeutics) database against protein targets from the **RCSB PDB**. It automates ligand library retrieval, receptor/ligand preparation, binding-site detection, multi-engine molecular docking, redocking validation, protein–ligand interaction analysis, and ADMET prediction — all from a single browser interface.

---

## ✨ Features

- **Automated phytochemical retrieval** — pulls compound names, SMILES, and PubChem CIDs directly from IMPPAT for any plant of interest.
- **Drug-likeness pre-filtering** — Lipinski's Rule of Five screening on the fly.
- **One-click receptor preparation** — waters, ions, glycans, cofactors, and buffer/cryoprotectant molecules stripped automatically; metal centers preserved with correct formal charges; polar hydrogens added at physiological pH; Gasteiger charges assigned; PDBQT-ready.
- **Automatic binding-site (grid box) detection** — centered on the co-crystallized ligand centroid, with manual override for apo structures.
- **pH-aware ligand preparation** — protonation state enumeration (Dimorphite-DL), 3D embedding (ETKDGv3), MMFF94/UFF energy minimization, Meeko/Open Babel PDBQT export.
- **Multi-engine docking** — AutoDock Vina, GNINA (CNN rescoring), and Uni-Dock (GPU-accelerated), run side-by-side for cross-validation.
- **Redocking / protocol validation** — automatic RMSD check against the native co-crystallized pose.
- **Interaction fingerprinting** — hydrogen bonds, hydrophobic contacts, and electrostatic/salt-bridge interactions computed geometrically per pose.
- **ADMET & drug-likeness reports** — RDKit-based physicochemical profiling, PAINS/Brenk structural alerts, and optional ADMET-AI predictions for top-ranked hits.
- **One-click export** — receptor/ligand files, docked poses, summary CSVs, and a full session report packaged into a downloadable ZIP.

---

## 🔬 Pipeline Overview

```
IMPPAT phytochemical retrieval
        │
        ▼
Lipinski / drug-likeness pre-filter
        │
        ▼
RCSB PDB structure retrieval  ──▶  Receptor cleanup & protonation (pH 7.4)
        │                                   │
        ▼                                   ▼
Co-crystal ligand detection  ──▶  Grid-box (binding site) definition
        │
        ▼
Ligand protonation (Dimorphite-DL) → 3D embedding (ETKDGv3) → MMFF/UFF minimization
        │
        ▼
PDBQT preparation (Meeko / Open Babel)
        │
        ▼
Molecular docking (Vina / GNINA / Uni-Dock)
        │
        ▼
Redocking RMSD validation  +  Pose ranking by binding affinity
        │
        ▼
Interaction analysis (H-bonds, hydrophobic, electrostatic)
        │
        ▼
ADMET / drug-likeness prediction (top hits)
        │
        ▼
Downloadable results (CSV, PDBQT, SDF, ZIP report)
```

---

## 🛠️ Requirements

### Python (≥ 3.9)
```
streamlit
requests
pandas
numpy
scipy
matplotlib
rdkit
prody
meeko
dimorphite-dl
admet-ai        # optional, for ML-based ADMET predictions
```

Install via:
```bash
pip install streamlit requests pandas numpy scipy matplotlib rdkit prody meeko dimorphite-dl
pip install admet-ai   # optional
```

### External binaries (must be installed and on PATH, or configured in-app)
| Tool | Purpose |
|---|---|
| [Open Babel](https://openbabel.org/) | Protonation, format conversion, Gasteiger charges |
| [AutoDock Vina](https://vina.scripps.edu/) | Molecular docking |
| [GNINA](https://github.com/gnina/gnina) | CNN-scoring docking (native binary or Docker) |
| [Uni-Dock](https://github.com/dptech-corp/Uni-Dock) | GPU-accelerated docking |
| Docker *(optional)* | Required only if running GNINA via container image |

> GNINA and Uni-Dock are optional — the app runs a fully functional AutoDock Vina workflow without them.

---

## 🚀 Installation

```bash
git clone https://github.com/<your-username>/IMPPATez.git
cd IMPPATez
pip install -r requirements.txt
```

Make sure `obabel`, `vina`, and (optionally) `gnina` / `unidock` are installed and discoverable on your system PATH.

---

## ▶️ Usage

```bash
streamlit run imppatez.py
```

Then, in the browser UI:

1. **Search** for a plant in IMPPAT or upload a custom SMILES/CSV list.
2. **Fetch** a PDB structure by ID or search RCSB directly.
3. Let the app **auto-detect** the co-crystallized ligand and binding site (or set the grid box manually).
4. Configure docking parameters (exhaustiveness, number of modes, energy range, pH) and choose an engine — **Vina**, **GNINA**, or **Uni-Dock**.
5. Run the docking job and review ranked results, interaction profiles, and (optionally) ADMET predictions for the top hits.
6. **Download** the full results package (ZIP) for your records or manuscript.

---

## 📁 Output

Each session produces:
- Prepared receptor (`.pdbqt`) and grid-box definition (`config.txt`, `box.pdb`)
- Prepared ligand structures (`.sdf`, `.pdbqt`) per compound
- Docked poses per compound/engine
- Per-ligand and summary results (`.csv`, `.txt`)
- ADMET summary (`.csv`)
- A single downloadable `.zip` session archive

---

## 📖 Methodology Summary

Receptor structures are cleaned (waters/ions/glycans/cofactors removed, metals charge-corrected), protonated at physiological pH via Open Babel, and converted to PDBQT with Gasteiger charges. The docking grid box is centered on the co-crystallized ligand's centroid (padding-based sizing, 12–24 Å per axis) or set manually for apo structures. Ligands are protonated at the target pH (Dimorphite-DL), embedded in 3D (ETKDGv3), energy-minimized (MMFF94/UFF), and converted to PDBQT (Meeko/Open Babel). Docking is performed with AutoDock Vina, GNINA, and/or Uni-Dock, with redocking-based RMSD validation where a native ligand pose is available. Poses are ranked by binding affinity, geometrically profiled for H-bond/hydrophobic/electrostatic interactions, and top hits are further assessed for drug-likeness and ADMET properties.

*For a full methods write-up suitable for manuscript submission, see [`METHODS.md`](METHODS.md) (or the Methods section of the associated paper).*

---

## 📚 Data & Tool Attribution

- **IMPPAT** — Indian Medicinal Plants, Phytochemistry And Therapeutics database ([cb.imsc.res.in/imppat](https://cb.imsc.res.in/imppat))
- **RCSB PDB** — Protein Data Bank structures
- **RDKit**, **Open Babel**, **Meeko**, **Dimorphite-DL**, **ProDy**
- **AutoDock Vina**, **GNINA**, **Uni-Dock**
- **ADMET-AI** *(optional)*

Please cite the original tool/database papers if you use this pipeline in published work.

---

## ⚠️ Disclaimer

This tool is intended for academic and research use in computational drug discovery. Docking scores and ADMET predictions are computational estimates and **do not replace experimental validation**.

---

## 👤 Author

Developed by **[Nabiul Orko](https://www.linkedin.com/in/mdnabiulhoque/)**

---

## 📄 License

Specify your license here (e.g., MIT, GPL-3.0, Apache-2.0).
